# VehiReID-RAG
VehiReID-RAG is a multimodal framework for vehicle re-identification, combining image captioning with retrieval-augmented generation. It extracts fine-grained visual details, generates descriptive captions, and queries a knowledge base to enable accurate cross-camera matching with grounded, interpretable results.

## How to use
Create a folder and clone the repo into it on your local computer/vm. Install the CompCars dataset from the Google Drive in the link https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt. Only install data files, ignore sv files. Please read the instructions on the stie on how to download. 
In your directory create a new folder called 'data' and within it another folder called compcars and place all the downloaded content into it. You should see data/compcars/ and 5 folders in there.
Create a virtual env and run ```bash pip install requirments.txt. Once everything is installed, run src/datasets/compcars_dataset.py and scripts/build_compcars_index.py. You should seedata/compcars/indexes created with two files. Then run torchrun --nproc_per_node=2 src/train.py --cfg ./config/default.yaml to start training. This will take a few hours. After training, you should see new foldder runs/. Then run torchrun --nproc_per_node=2 src/eval.py --ckpt ./runs/best.pt for evaluation and you should get the metircs to record. After that you can download any car image from google and run predict.py on it to see how the model performs. I did: python src/predict.py --ckpt runs/best.pt --image carimage.jpg

