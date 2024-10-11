
# Download the images

# EXtract the image

cd ./Truck

# mkdir images

# for i in $(ls *.jpg); do
#     mv $i images
# done

# echo "Extracting features..."
# colmap feature_extractor --database_path ./database.db --image_path ./images --ImageReader.single_camera 1 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1

# echo "Running feature matching..."
# colmap exhaustive_matcher --database_path ./database.db --SiftMatching.max_distance 1

# echo "Reconstructing 3D model..."
# colmap mapper --database_path ./database.db --image_path ./images --output_path ./

# echo "Done!"
# colmap gui --import_path ./0 --database_path ./database.db --image_path ./images

ns-process-data images --skip-colmap --data ./images --colmap-model-path ./0 --output_dir ./