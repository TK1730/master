import argparse
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def visualize_logs(log_path, output_dir):
    # Find event files
    event_files = []
    if os.path.isfile(log_path):
        event_files.append(log_path)
    elif os.path.isdir(log_path):
        for root, dirs, files in os.walk(log_path):
            for file in files:
                if "tfevents" in file:
                    event_files.append(os.path.join(root, file))
    else:
        print(f"Error: {log_path} is not a valid file or directory")
        return

    if not event_files:
        print(f"No tfevents files found in {log_path}")
        return

    print(f"Found {len(event_files)} event files.")
    
    for event_file in event_files:
        print(f"Processing {event_file}...")
        
        # Create a specific output directory for this file
        file_name = os.path.basename(event_file)
        # Verify allow valid directory name
        run_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(run_output_dir, exist_ok=True)
        
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()

            # Initialize dictionary to store data for this file
            data = {}
            
            # Check if scalars are available
            if 'scalars' not in ea.Tags():
                print(f"No scalars found in {event_file}")
                continue

            tags = ea.Tags()['scalars']
            
            for tag in tags:
                events = ea.Scalars(tag)
                if tag not in data:
                    data[tag] = {'steps': [], 'values': []}
                
                for event in events:
                    data[tag]['steps'].append(event.step)
                    data[tag]['values'].append(event.value)

            # Sort and Plot
            for tag, values in data.items():
                if not values['steps']:
                    continue
                    
                steps = values['steps']
                vals = values['values']
                sorted_pairs = sorted(zip(steps, vals))
                steps = [x[0] for x in sorted_pairs]
                vals = [x[1] for x in sorted_pairs]

                plt.figure(figsize=(10, 6))
                plt.plot(steps, vals, label=tag)
                plt.xlabel('Steps')
                plt.ylabel('Value')
                plt.title(f'{tag} over steps')
                plt.legend()
                plt.grid(True)
                
                # Replace slashes in tag name for filename
                safe_tag = tag.replace('/', '_')
                output_path = os.path.join(run_output_dir, f"{safe_tag}.png")
                plt.savefig(output_path)
                plt.close()
                print(f"  Saved plot for {tag} to {output_path}")

        except Exception as e:
            print(f"Failed to process {event_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize TensorBoard logs")
    parser.add_argument("--log_path", type=str, default="output/tensorboard", help="Path to TensorBoard log file or directory")
    parser.add_argument("--output_dir", type=str, default="output/plots", help="Path to save plots")
    
    args = parser.parse_args()
    
    visualize_logs(args.log_path, args.output_dir)
