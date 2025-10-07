import os

# Define the root directory and output text file
root_dir = "Audio/source_separation/train"
output_file = "dataset/file_paths_enhancement.txt"

# Open the output file
with open(output_file, "w") as f:
    # Iterate through the folders in the root directory
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        
        if os.path.isdir(folder_path):
            # Find the paths for signal_mix and voice_mix files
            mix_path = None
            voice_path = None
            
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if "mix_snr" in file_name:
                    mix_path = file_path
                elif "voice" in file_name:
                    voice_path = file_path

                    
            
            # Ensure both paths are found before writing to the file
            if mix_path and voice_path:
                f.write(f"{mix_path} {voice_path} \n")

print(f"Paths saved to {output_file}")