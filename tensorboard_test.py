
import os
import glob
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return [sum(data[i:i + window_size]) / window_size for i in range(len(data) - window_size + 1)]


def extract_scalars_from_event_files(log_dir, target_tags):
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"❌ No .tfevents files found in: {log_dir}")
        sys.exit(1)

    print(f"✅ Found {len(event_files)} log files.")
    tag_data = {tag: [] for tag in target_tags}

    for file in sorted(event_files):
        print(f"📘 Reading: {file}")
        ea = EventAccumulator(file)
        ea.Reload()

        for tag in target_tags:
            if tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                tag_data[tag].extend((event.step, event.value) for event in events)
            else:
                print(f"⚠️ Tag '{tag}' not found in {file}")

    valid_tags = [tag for tag, data in tag_data.items() if data]
    if not valid_tags:
        print("❌ No valid scalar tags found.")
        sys.exit(1)

    return tag_data, valid_tags


def plot_scalars(tag_data, valid_tags, output_path, smooth=False):
    plt.figure(figsize=(14, 4))

    for i, tag in enumerate(valid_tags):
        plt.subplot(1, len(valid_tags), i + 1)

        tag_data_sorted = sorted(tag_data[tag], key=lambda x: x[0])
        steps, values = zip(*tag_data_sorted)

        if smooth:
            values = moving_average(values)
            steps = steps[:len(values)]

        plt.plot(steps, values, label=tag.split("/")[-1])
        plt.xlabel("Step")
        plt.ylabel(tag.split("/")[-1])
        plt.title(f"{tag} (smoothed)" if smooth else tag)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved plot to {output_path}")
    plt.show()


def main():
    logdir = "checkpoints/example_lora/tblog"
    output_path = "training_metrics.png"
    smooth = True

    target_tags = ["Loss/train", "LearningRate", "GradientNorm"]
    tag_data, valid_tags = extract_scalars_from_event_files(logdir, target_tags)
    plot_scalars(tag_data, valid_tags, output_path, smooth)


if __name__ == "__main__":
    main()
