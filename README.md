Robotic Sequencer

This repository is an extension of the paper "Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation"(Paper, Website, Presentation) by Yuanpei Chen*, Chen Wang*, Li Fei-Fei and C. Karen Liu. 

Installation
python 3.8
conda create -n roboseq python=3.8
conda activate roboseq
IsaacGym (tested with Preview Release 3/4 and Preview Release 4/4). Follow the instruction to download the package.
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
cd examples
(test installation) python examples/joint_monkey.py

pip install -r requirements.txt
pip install -e .

Training
Taking the Block Assembly as an example, each sub-task and their order are BlockAssemblySearch->BlockAssemblyOrient->BlockAssemblyGraspSim->BlockAssemblyInsertSim.

If you want to use the bi-directional optimization in our paper to train the BlockAssembly or ToolPositioning task, simply:

python scripts/bi-optimization.py --task=[BlockAssembly / ToolPositioning]
Since each sub-task takes about 1~2 days to train, it may take a long time for the whole process, so we also provide a way to train each sub-task individually. When training each sub-task, run this line in dexteroushandenvs folder:

python train_rlgames.py --task=[BlockAssemblySearch / BlockAssemblyOrient / BlockAssemblyGraspSim / BlockAssemblyInsertSim]   --num_envs=1024
The trained model will be saved to runs folder, and the terminal state of the task will be saved to immediate_state. These terminal states are also used to train the transition feasibility function (see our paper), using the following command:

python policy_sequencing/tvalue_trainer.py --task=[BlockAssemblySearch / BlockAssemblyOrient / BlockAssemblyGraspSim / BlockAssemblyInsertSim]
Evaluation
To load a trained model and only perform inference (no training) in each sub-task, pass --play as an argument, and pass --checkpoint to specify the trained models which you want to load. Here is an example in BlockAssemblyGraspSim task:

python train_rlgames.py --task=BlockAssemblyGraspSim  --checkpoint=./checkpoint/block_assembly/last_AllegroHandLegoTestPAISim_ep_19000_rew_1530.9819.pth --play --num_envs=256
During evaluation in our paper, infer each sub-policy in the same order and calculate the overall success rate at the last task. Simply

python scripts/evaluation.py
It would save the terminal state of each sub-task in inference and finally count the success rate in the Insertion task.

Acknowledgement
Contributors from Bi-DexHands and Sequential Dexterity.
