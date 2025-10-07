import os

# Define the root directory and output text file
root_dir = "Audio/source_separation/test"
output_file = "dataset/file_paths_test_infer.txt"

root_dir_path = os.path.abspath(root_dir)
output_file = os.path.abspath(output_file)

# Open the output file
with open(output_file, "w") as f:
    # Iterate through the folders in the root directory
    for folder in sorted(os.listdir(root_dir_path)):
        folder_path = os.path.join(root_dir, folder)
        
        if os.path.isdir(folder_path):
            # Find the paths for signal_mix and voice_mix files
            mix_path = None
            voice_path = None
            noise_path = None
            
            for file_name in os.listdir(folder_path):
                if "mix_snr" in file_name:
                    mix_path = os.path.join(root_dir ,folder, file_name)

                    
            # Ensure all paths are found before writing to the file
            if mix_path :
                f.write(f"{mix_path}\n")

print(f"Paths saved to {output_file}")
