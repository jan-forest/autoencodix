
# Instructions for the GA Workshop 
## Option A: Working in Home Dir:

1. Set up VPN and instructed by a document provided via email.

2. Open JupyterLab and clone the AUTOENCODIX git repository via the link:
https://jupyterhub.hpc.tu-dresden.de/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fjan-forest%2Fautoencodix&urlpath=tree%2Fautoencodix%2F&branch=main#/~(cluster~'capella~nodes~'1~ntasks~'1~cpuspertask~'4~mempercpu~'2048~gres~'gpu*3a1~runtime~'06*3a00*3a00~project~'p_scads_trainings)
### Spawn resources need to be edited before the start

- Click on JupyterLab
- Navigate to autoencodix in the directory on the left side and open install-kernel-script.ipynb. 
- Run install-kernel-script.ipynb
- Follow the instructions indicated in the output. 

### Troubleshooting: 
If you are running on MacOS and encountering workspace creation issues, please see [here](https://doc.zih.tu-dresden.de/data_lifecycle/workspaces/?h=workspace#faq-and-troubleshooting)

## Option B: Working in a Workspace:
1. Set up VPN and instructed by a document provided via email. 

2. Open JupyterLab via the following link:
https://jupyterhub.hpc.tu-dresden.de/hub/spawn#/~(cluster~'capella~nodes~'1~ntasks~'1~cpuspertask~'4~mempercpu~'2048~gres~'gpu*3a1~runtime~'06*3a00*3a00~project~'p_scads_trainings)
### Spawn resources need to be edited before the start   

3. Open Terminal

4. Type in the following commands:
```
ws_allocate -F horse AE-ws 10
ws_register -F horse $HOME
cd $HOME/horse/$USER-AE-ws/
git clone -b ga-workshop https://github.com/jan-forest/autoencodix
```

5. Navigate to horse/autoencodix in the directory view on the left side and open install-kernel-script.ipynb.

6. Run install-kernel-script.ipynb 

7. Follow the instructions indicated in the output. 

### Troubleshooting:
If you are running on MacOS and encountering workspace creation issues, please see [here](https://doc.zih.tu-dresden.de/data_lifecycle/workspaces/?h=workspace#faq-and-troubleshooting) 
