Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1

su DGX permette di modificare il container sanboxato  ma senza nvidia-smi
apptainer shell --nv --fakeroot -u -w xvbf-hub/

apt-get install libnvidia-egl-wayland1

da fare andare in windows base init e inserire print, l'idea è che gli da in input lo schermo sbagliato.


apt-get install libnvidia-gl-470-server ??? funziona solo se non rischiedo nvidia-smi



Su orfeoDGX funzionante --------------------

apptainer run --nv xvbf-hub/

source /opt/miniconda/bin/activate
conda activate sofa

cd xvbf-hub/app/sofa_zoo/

python3 sofa_zoo/envs/grasp_lift_touch/ppo.py


27pfs 8 env
27pfs 16 env

----------------------------------------

leonardo boost_usr prod debug

singularity run --nv xvbf-hub/
