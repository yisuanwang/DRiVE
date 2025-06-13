#!/bin/bash
# source path to your/miniconda3/etc/profile.d/conda.sh
# conda activate drive 

# ====================== Select Input Dataset ======================= #
is_text=false # Input data is text

# # Any-pose Real Human -> T-pose Real Human -> T-pose Anime -> 3DGS
# testdataset_name=Anypose-real2anime
# input_image_dir=./input/$testdataset_name # Path to image files, needs to be modified
# is_Anime=false # Input data is not T-pose Anime
# is_Tpose=false # Set to true if input data is T-pose Anime, sd-webui is not needed
# to_Anime=true # Whether to finally convert to anime

# # T-pose Real Human -> T-pose Anime -> 3DGS
# testdataset_name=Tpose-real2anime
# input_image_dir=./input/$testdataset_name # Path to image files, needs to be modified
# is_Anime=false # Input data is not T-pose Anime
# is_Tpose=true # Set to true if input data is T-pose Anime, sd-webui is not needed
# to_Anime=true

# Any-pose Anime -> T-pose Anime -> 3DGS
# testdataset_name=Anypose-anime
# input_image_dir=./input/$testdataset_name # Path to image files, needs to be modified
# is_Anime=true # Input data is T-pose Anime
# is_Tpose=false # Input data is T-pose Anime
# to_Anime=true

# T-pose Anime -> 3DGS
testdataset_name=Tpose-anime
input_image_dir=./input/$testdataset_name # Path to image files, needs to be modified
is_Anime=true # Input data is T-pose Anime
is_Tpose=true # Input data is T-pose Anime
to_Anime=true

# ====================== input dataset ======================= #
image_paths_file=$input_image_dir/img_path_list.txt
sdwebui_port=7888 # sdwebui port
sd_webui_dir=/cpfs04/shared/IDC_dongjunting_group/cjh/project/DRiVE/sub/stable-diffusion-webui
root_dir=/cpfs04/shared/IDC_dongjunting_group/cjh/project/DRiVE
use_vqa=false 


# make image path list
mkdir -p $input_image_dir
python ./scripts/makedatasetpath.py \
    --directory $input_image_dir \
    --output_file $image_paths_file


if [ ! -f "$image_paths_file" ]; then
    echo "File not found: $image_paths_file"
    exit 1
fi

if ! { { [ "$is_Anime" = false ] && [ "$is_Tpose" = true ] && [ "$to_Anime" = false ]; } || { [ "$is_Anime" = true ] && [ "$is_Tpose" = true ] && [ "$to_Anime" = true ]; } }; then
  echo "[run sd-webui...]"
  PID=$(sudo netstat -tulnp | grep ":$sdwebui_port" | awk '{print $7}' | cut -d'/' -f1)
  if [ -z "$PID" ]; then
    echo "no PID in port $sdwebui_port"
  else
    echo "find PID=$PID in port $sdwebui_port, try to stop"
    sudo kill $PID
    if [ $? -eq 0 ]; then
      echo "PID=$PID stopped"
    else
      echo "stop $PID error"
    fi
  fi
  LOG_FILE=$root_dir/webui.log
  echo " " > "$LOG_FILE"
  cd $sd_webui_dir
  nohup bash ./webui.sh --port $sdwebui_port > $LOG_FILE 2>&1 &
  SEARCH_STRING="Applying attention optimization: Doggettx... done."

  cd $root_dir

  if [ ! -f "$LOG_FILE" ]; then
      echo "log error: $LOG_FILE"
      exit 1
  fi
  while true; do
      if grep -q "$SEARCH_STRING" "$LOG_FILE"; then
          echo "run sd-webui successfully!"
          break
      else
          echo "wait to start sd-webui..."
          sleep 3
      fi
  done
fi
# ======= webui start =======


while IFS= read -r line
do
    image_path=$(echo "$line" | xargs) 
    scripts_path=./scripts/pipeline.sh # real human anypose

    if [ "$is_Anime" = true ] && [ "$is_Tpose" = true ] && [ "$to_Anime" = true ]; then 
        # Tpose Anime -> 3DGS
        scripts_path=./scripts/pipeline_from_TposeAnime.sh
    elif [ "$is_Anime" = true ] && [ "$is_Tpose" = false ] && [ "$to_Anime" = true ]; then
        # Anypose Anime -> Tpose Anime -> 3DGS
        scripts_path=./scripts/pipeline_from_AnyposeAnime.sh
    elif [ "$is_Anime" = false ] && [ "$is_Tpose" = true ] && [ "$to_Anime" = true ]; then
        # Tpose real -> Tpose Anime -> 3DGS
        scripts_path=./scripts/pipeline_from_TposeHuman.sh
    elif [ "$is_Anime" = false ] && [ "$is_Tpose" = true ] && [ "$to_Anime" = false ]; then
        # Tpose real -> 3DGS
        scripts_path=./scripts/pipeline_from_TposeHuman2Human.sh
    elif [ "$is_Anime" = false ] && [ "$is_Tpose" = false ] && [ "$to_Anime" = true ]; then
        # Anypose real -> Tpose真人 -> Tpose Anime -> 3DGS
        scripts_path=./scripts/pipeline.sh # real human anypose
    fi

    echo "run script: $scripts_path"

    if bash $scripts_path $image_path $sdwebui_port $use_vqa $testdataset_name; then
        echo "Command executed successfully for: $image_path"
    else
        echo "An error occurred while executing the command for: $image_path"
    fi

done < "$image_paths_file"

# ======= webui end ========
echo "[kill sd-webui...]"
PORT=$sdwebui_port
PID=$(sudo netstat -tulnp | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1)


if [ -z "$PID" ]; then
  echo "no PID in port $PORT"
else
  echo "find port $PORT, PID= $PID"
  sudo kill $PID

  if [ $? -eq 0 ]; then
    echo "$PID stop"
  else
    echo "stop $PID error"
  fi
fi
