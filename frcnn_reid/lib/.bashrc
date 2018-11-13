# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# added by Anaconda3 installer
#export PATH="/home/zxdong/anaconda3/bin:$PATH"
#export PATH="/home/zhuxudong/anaconda3/bin:$PATH"

# added by Anaconda3 installer
export PATH="/home/zhuxudong/anaconda3/bin:$PATH"
export PATH="$PATH:$HOME/.local/bin:$HOME/bin:/usr/local/cuda-9.0/bin"
LIBRARY_PATH=$LIBRARY_PATH:$HOME/lib

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64:$HOME/lib
CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HOME/NVIDIA_CUDA-9.0_Samples/common/inc:$HOME/include

export LIBRARY_PATH
export LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH
