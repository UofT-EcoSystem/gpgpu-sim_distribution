docker build . -t gpusim --build-arg user=$USER --build-arg userid=`id -u` && \
docker run --privileged --hostname docker.gpusim -dit \
  -e DISPLAY=$DISPLAY \
  -v `pwd`/../../:/mnt --ipc=host --name gpusim gpusim
