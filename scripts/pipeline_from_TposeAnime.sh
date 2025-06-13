# from T pose Anime 

input_prompt=''
input_image_path=$1 # as Anime0.png
sdwebui_port=$2
use_vqa=$3  # vqa, defauls false
testdataset_name=$4

echo "===== run pipeline at picture : $input_image_path ====="
# echo "Image Path: $input_image_path"
echo "SDWebUI Port: $sdwebui_port"
echo "Use VQA: $use_vqa"
echo "testdataset_name: $testdataset_name"

filename_with_extension=$(basename "$input_image_path")
filename_without_extension="${filename_with_extension%.*}"


workspace=./output/$testdataset_name/$(basename "$filename_without_extension")
echo  "---- workspace at : $workspace ----"

rm -r $workspace
mkdir -p $workspace


echo "[step 1: crop & rmbg]"
python anime-segmentation/inferimg.py --input $input_image_path --output $workspace/Anime.png


echo "[step 5: crop head]"

python tool/crophead/crop.py $workspace/Anime.png $workspace/Anime-head.png
python tool/crophead/crop-body.py $workspace/Anime.png $workspace/Anime-body.png


echo "[step 6: LGM]"
cd LGM
mkdir ../$workspace/lgm-full
mkdir ../$workspace/lgm-head

python infer.py big \
    --resume path_to_your/lgm-model-512_512_0426.safetensors \
    --workspace ../$workspace/lgm-full \
    --test_path ../$workspace/Anime.png
