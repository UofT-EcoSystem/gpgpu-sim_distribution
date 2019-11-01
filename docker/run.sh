docker build . -t gpusim
docker run --privileged --hostname docker.gpusim -dit \
  -e DISPLAY=$DISPLAY \
  --runtime=nvidia -v `pwd`/../../:/mnt --ipc=host --name gpusim gpusim
