docker run --privileged -dit -h docker.example.com -p 10023:22 \
  --ipc=host --runtime=nvidia \ 
  -v /home/serinatan/project:/mnt \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --name sim-aerial gpusim90:aerial
