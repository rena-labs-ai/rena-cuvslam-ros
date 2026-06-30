// Shared primitives for cuVSLAM tracker implementations.
//
// LatestSlot<T>:     single-slot latest-wins mailbox used by both RGBD and
//                    Stereo trackers to decouple DDS ingestion from Track().
// CameraStatsLogger: per-second per-camera diagnostic ladder — same one-line
//                    format as the Python node so the two nodes' logs are
//                    directly comparable.
//
// All methods are inline so this header has no companion .cpp and can be
// included by multiple translation units without ODR violations.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "rena_cuvslam_ros/frame_conversions.hpp"

namespace rena_cuvslam {

using ResultCallback =
    std::function<void(int64_t timestamp_ns, const RosPose& pose)>;

// ---------------------------------------------------------------------------
// LatestSlot<T>
// Single-slot latest-wins mailbox: put() overwrites whatever is pending so a
// slow track stage never stalls ingestion; get() blocks until an item or close.
// ---------------------------------------------------------------------------
template <typename T>
class LatestSlot {
 public:
  inline void put(T item) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      item_ = std::move(item);
      has_ = true;
    }
    cv_.notify_one();
  }

  // Returns false on timeout or after close() with no pending item.
  inline bool get(T& out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!has_ && !closed_)
      cv_.wait_for(lk, timeout, [&] { return has_ || closed_; });
    if (!has_) return false;
    out = std::move(item_);
    has_ = false;
    return true;
  }

  inline void close() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      closed_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  T item_{};
  bool has_ = false;
  bool closed_ = false;
};

// ---------------------------------------------------------------------------
// CameraStatsLogger
// Per-second per-camera + per-stage diagnostic ladder. One-line format:
//   [<tag>] cam0 rgb 30 depth 30 | cam1 rgb 30 depth 29 |
//       sync 30/s | decode 30/s | track 28/s <=33 26 >33 2 (sum 90ms)
// For stereo trackers "rgb" = left frames, "depth" = right frames.
// ---------------------------------------------------------------------------
class CameraStatsLogger {
 public:
  static constexpr double kBudgetMs = 1000.0 / 30.0;  // ~33.3 ms

  CameraStatsLogger(const rclcpp::Logger& logger, std::string tag,
                    int n_cameras, bool debug)
      : logger_(logger), tag_(std::move(tag)), n_(n_cameras), debug_(debug) {
    raw_rgb_.assign(n_cameras, 0);
    raw_depth_.assign(n_cameras, 0);
    last_log_ = std::chrono::steady_clock::now();
  }

  inline void record_raw(int cam, bool is_depth) {
    std::lock_guard<std::mutex> lk(mu_);
    if (is_depth)
      ++raw_depth_[cam];
    else
      ++raw_rgb_[cam];
    maybe_log_locked();
  }

  inline void record_sync() {
    std::lock_guard<std::mutex> lk(mu_);
    ++sync_n_;
  }

  inline void record_decode() {
    std::lock_guard<std::mutex> lk(mu_);
    ++decode_n_;
  }

  inline void record_track(double track_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    ++track_n_;
    if (track_ms > kBudgetMs) {
      ++over_n_;
      over_ms_ += track_ms;
    } else {
      ++on_n_;
    }
  }

 private:
  inline void maybe_log_locked() {
    // TEMP: always emit the one-line ladder (INFO, 1/s) regardless of `debug_`
    // so we don't have to set global log_level:=debug (which spams everything).
    (void)debug_;
    using namespace std::chrono_literals;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log_ < 1s) return;

    std::string cams;
    for (int i = 0; i < n_; ++i) {
      if (i) cams += " | ";
      cams += "cam" + std::to_string(i) + " rgb " + std::to_string(raw_rgb_[i]) +
              " depth " + std::to_string(raw_depth_[i]);
    }
    RCLCPP_INFO(
        logger_,
        "[%s] %s | sync %d/s | decode %d/s | track %d/s <=33 %d >33 %d (sum %.0fms)",
        tag_.c_str(), cams.c_str(), sync_n_, decode_n_, track_n_, on_n_, over_n_,
        over_ms_);

    std::fill(raw_rgb_.begin(), raw_rgb_.end(), 0);
    std::fill(raw_depth_.begin(), raw_depth_.end(), 0);
    sync_n_ = decode_n_ = track_n_ = on_n_ = over_n_ = 0;
    over_ms_ = 0.0;
    last_log_ = now;
  }

  rclcpp::Logger logger_;
  std::string tag_;
  int n_;
  bool debug_;
  std::mutex mu_;
  std::vector<int> raw_rgb_, raw_depth_;
  int sync_n_ = 0, decode_n_ = 0, track_n_ = 0, on_n_ = 0, over_n_ = 0;
  double over_ms_ = 0.0;
  std::chrono::steady_clock::time_point last_log_;
};

}  // namespace rena_cuvslam
