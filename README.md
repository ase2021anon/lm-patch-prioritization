# Language Model Patch Prioritzation

Code for the ASE 2021 submission [Language Models Can Prioritize Patches for Practical Program Patching]

## Requirements
 
 * Python 3.7
 * [srcML](https://www.srcml.org/) 1.0
 
## Usage
 
### Setup
 1. Decompress the `patch_data/tool_patches.tar.gz` file, which has generated patches from 5 APR tools. The decompressed size is 773MB.
 2. The srcML files for all the patches must be generated before the next step. An example script that does this for the TBar-generated tools is provided in `patch_processing.sh`. 
 3. Get the pretrained model from this [Zenodo link](https://zenodo.org/record/4717485).
 4. Edit the `order_by_nat.sh` script parameters; for example, make sure the `XML_DIR` is pointing toward the directory in which the srcML XML files reside, among others. Look to the script comments for descriptions of each parameter.
 5. `pip install -r requirements.txt`
 
### Execution
```
sh order_by_nat.sh <PROJECT> <BUG_NUM>
``` 

should generate a visualized list of the language model's calculations in the `visualizations` directory.
