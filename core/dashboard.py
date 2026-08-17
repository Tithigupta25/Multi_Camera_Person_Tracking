import os
import numpy as np
import matplotlib.pyplot as plt


def create_dashboard(camera_data, output_path):

    # =========================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    if not camera_data:
        print("[WARNING] No camera data available for dashboard.")
        return

    cameras = sorted(camera_data.keys())

    # =========================================================
    # EXTRACT METRICS
    # =========================================================

    frames_pushed = []
    frames_dropped = []
    drop_rates = []

    max_queue_fullness = []
    avg_queue_fullness = []

    avg_processing_lag = []
    max_processing_lag = []

    detections = []

    unique_tracks = []
    active_tracks = []

    for camera_id in cameras:

        data = camera_data[camera_id]

        # -----------------------------------------------------
        # FRAME METRICS
        # -----------------------------------------------------

        frames_pushed.append(
            data.get("frames_pushed", 0)
        )

        frames_dropped.append(
            data.get("frames_dropped", 0)
        )

        drop_rates.append(
            data.get("drop_rate", 0)
        )

        # -----------------------------------------------------
        # QUEUE METRICS
        # -----------------------------------------------------

        queue_values = data.get(
            "queue_fullness",
            []
        )

        if queue_values:

            max_queue_fullness.append(
                max(queue_values)
            )

            avg_queue_fullness.append(
                sum(queue_values) / len(queue_values)
            )

        else:

            max_queue_fullness.append(0)
            avg_queue_fullness.append(0)

        # -----------------------------------------------------
        # PROCESSING LAG
        # -----------------------------------------------------

        lag_values = data.get(
            "processing_lags",
            []
        )

        if lag_values:

            avg_processing_lag.append(
                sum(lag_values) / len(lag_values)
            )

            max_processing_lag.append(
                max(lag_values)
            )

        else:

            avg_processing_lag.append(0)
            max_processing_lag.append(0)

        # -----------------------------------------------------
        # DETECTIONS
        # -----------------------------------------------------

        detections.append(
            data.get("detections", 0)
        )

        # -----------------------------------------------------
        # TRACKING
        # -----------------------------------------------------

        unique_tracks.append(
            data.get("unique_tracks", 0)
        )

        active_tracks.append(
            data.get("active_tracks", 0)
        )

    # =========================================================
    # CONVERT VALUES FOR DISPLAY
    # =========================================================

    camera_labels = [
        f"Camera {camera}"
        for camera in cameras
    ]

    drop_rates_percent = [
        value * 100
        for value in drop_rates
    ]

    max_queue_percent = [
        value * 100
        for value in max_queue_fullness
    ]

    avg_queue_percent = [
        value * 100
        for value in avg_queue_fullness
    ]

    # =========================================================
    # PIPELINE HEALTH
    # =========================================================

    health_values = []
    health_labels = []

    for i in range(len(cameras)):

        drop_rate = drop_rates[i]
        queue_fullness = max_queue_fullness[i]

        if (
            drop_rate >= 0.50
            or
            queue_fullness >= 0.95
        ):

            health_values.append(3)
            health_labels.append("Critical")

        elif (
            drop_rate >= 0.20
            or
            queue_fullness >= 0.80
        ):

            health_values.append(2)
            health_labels.append("Warning")

        else:

            health_values.append(1)
            health_labels.append("Healthy")

    # =========================================================
    # GLOBAL SUMMARY
    # =========================================================

    total_pushed = sum(frames_pushed)
    total_dropped = sum(frames_dropped)
    total_detections = sum(detections)
    total_unique_tracks = sum(unique_tracks)

    overall_drop_rate = (
        total_dropped / total_pushed
        if total_pushed > 0
        else 0
    )

    all_lags = []

    for camera_id in cameras:

        all_lags.extend(
            camera_data[camera_id].get(
                "processing_lags",
                []
            )
        )

    overall_avg_lag = (
        sum(all_lags) / len(all_lags)
        if all_lags
        else 0
    )

    # =========================================================
    # CREATE FIGURE
    # =========================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 13)
    )

    # =========================================================
    # 1. PIPELINE HEALTH
    # =========================================================

    axes[0, 0].bar(
        camera_labels,
        health_values
    )

    axes[0, 0].set_title(
        "Pipeline Health"
    )

    axes[0, 0].set_ylabel(
        "Health Level"
    )

    axes[0, 0].set_yticks(
        [1, 2, 3]
    )

    axes[0, 0].set_yticklabels(
        [
            "Healthy",
            "Warning",
            "Critical"
        ]
    )

    axes[0, 0].tick_params(
        axis="x",
        rotation=20
    )

    # =========================================================
    # 2. FRAME DROP RATE
    # =========================================================

    axes[0, 1].bar(
        camera_labels,
        drop_rates_percent
    )

    axes[0, 1].set_title(
        "Frame Drop Rate"
    )

    axes[0, 1].set_ylabel(
        "Drop Rate (%)"
    )

    axes[0, 1].set_ylim(
        0,
        max(100, max(drop_rates_percent) + 10)
    )

    axes[0, 1].tick_params(
        axis="x",
        rotation=20
    )

    # =========================================================
    # 3. QUEUE FULLNESS
    # =========================================================

    x = np.arange(
        len(cameras)
    )

    width = 0.35

    axes[0, 2].bar(
        x - width / 2,
        avg_queue_percent,
        width,
        label="Average"
    )

    axes[0, 2].bar(
        x + width / 2,
        max_queue_percent,
        width,
        label="Maximum"
    )

    axes[0, 2].set_title(
        "Queue Fullness"
    )

    axes[0, 2].set_ylabel(
        "Queue Usage (%)"
    )

    axes[0, 2].set_ylim(
        0,
        105
    )

    axes[0, 2].set_xticks(
        x
    )

    axes[0, 2].set_xticklabels(
        camera_labels,
        rotation=20
    )

    axes[0, 2].legend()

    # =========================================================
    # 4. PROCESSING LATENCY
    # =========================================================

    axes[1, 0].bar(
        camera_labels,
        avg_processing_lag
    )

    axes[1, 0].set_title(
        "Average Processing Latency"
    )

    axes[1, 0].set_ylabel(
        "Latency (seconds)"
    )

    axes[1, 0].tick_params(
        axis="x",
        rotation=20
    )

    # =========================================================
    # 5. FRAMES PUSHED VS DROPPED
    # =========================================================

    axes[1, 1].bar(
        x - width / 2,
        frames_pushed,
        width,
        label="Pushed"
    )

    axes[1, 1].bar(
        x + width / 2,
        frames_dropped,
        width,
        label="Dropped"
    )

    axes[1, 1].set_title(
        "Frames Pushed vs Dropped"
    )

    axes[1, 1].set_ylabel(
        "Frames"
    )

    axes[1, 1].set_xticks(
        x
    )

    axes[1, 1].set_xticklabels(
        camera_labels,
        rotation=20
    )

    axes[1, 1].legend()

    # =========================================================
    # 6. DETECTIONS
    # =========================================================

    axes[1, 2].bar(
        camera_labels,
        detections
    )

    axes[1, 2].set_title(
        "Total Detections"
    )

    axes[1, 2].set_ylabel(
        "Detections"
    )

    axes[1, 2].tick_params(
        axis="x",
        rotation=20
    )

    # =========================================================
    # 7. TRACKING
    # =========================================================

    if any(unique_tracks):

        axes[2, 0].bar(
            x - width / 2,
            unique_tracks,
            width,
            label="Unique Tracks"
        )

        axes[2, 0].bar(
            x + width / 2,
            active_tracks,
            width,
            label="Active Tracks"
        )

        axes[2, 0].set_title(
            "Object Tracking"
        )

        axes[2, 0].set_ylabel(
            "Objects"
        )

        axes[2, 0].set_xticks(
            x
        )

        axes[2, 0].set_xticklabels(
            camera_labels,
            rotation=20
        )

        axes[2, 0].legend()

    else:

        axes[2, 0].text(
            0.5,
            0.5,
            "Tracking metrics\nnot available",
            ha="center",
            va="center",
            fontsize=12
        )

        axes[2, 0].set_title(
            "Object Tracking"
        )

        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])

    # =========================================================
    # 8. MAXIMUM PROCESSING LAG
    # =========================================================

    axes[2, 1].bar(
        camera_labels,
        max_processing_lag
    )

    axes[2, 1].set_title(
        "Maximum Processing Lag"
    )

    axes[2, 1].set_ylabel(
        "Lag (seconds)"
    )

    axes[2, 1].tick_params(
        axis="x",
        rotation=20
    )

    # =========================================================
    # 9. SYSTEM SUMMARY
    # =========================================================

    axes[2, 2].axis(
        "off"
    )

    summary_text = (
        "PIPELINE SUMMARY\n"
        "────────────────────────\n\n"
        f"Cameras: {len(cameras)}\n\n"
        f"Frames Pushed: {total_pushed}\n"
        f"Frames Dropped: {total_dropped}\n"
        f"Overall Drop Rate: "
        f"{overall_drop_rate * 100:.2f}%\n\n"
        f"Total Detections: "
        f"{total_detections}\n\n"
        f"Unique Tracks: "
        f"{total_unique_tracks}\n\n"
        f"Avg Processing Lag: "
        f"{overall_avg_lag:.3f} sec"
    )

    axes[2, 2].text(
        0.05,
        0.95,
        summary_text,
        transform=axes[2, 2].transAxes,
        fontsize=12,
        verticalalignment="top",
        family="monospace"
    )

    # =========================================================
    # MAIN TITLE
    # =========================================================

    fig.suptitle(
        "Multi-Camera Video Analytics Pipeline",
        fontsize=20,
        fontweight="bold"
    )

    # =========================================================
    # FOOTER
    # =========================================================

    fig.text(
        0.5,
        0.01,
        "Detection • Tracking • Queue Monitoring • Performance",
        ha="center",
        fontsize=10
    )

    # =========================================================
    # SAVE
    # =========================================================

    plt.tight_layout(
        rect=[
            0,
            0.03,
            1,
            0.95
        ]
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Dashboard saved to: {output_path}"
    )