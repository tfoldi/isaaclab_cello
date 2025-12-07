# **HCLTech-Cello-Reach-v0 Project**

## **🚀 Overview**

This repository contains the project for the HCLTech-Cello-Reach-v0 task, built as an extension based on Isaac Lab.  
It allows for isolated development and testing of the specific robotic reach task, outside of the core Isaac Lab repository.  
**Key Features:**

* Isolation Develop the **HCLTech-Cello-Reach-v0** task logic outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.  
* Flexibility This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, isaaclab, **hcltech, cello, reach**

## **🛠️ Installation**

* Install Isaac Lab by following the installation guide.  
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.  
* Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the IsaacLab directory):  
* Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:  
  \# use 'PATH\_TO\_isaaclab.sh|bat \-p' instead of 'python' if Isaac Lab is not installed in Python venv or conda  
  python \-m pip install \-e source/hcltech\_cello\_reach

  *(Assuming your extension package is named hcltech\_cello\_reach)*  
* Verify that the extension is correctly installed by:  
  * Listing the available tasks:  
    Note: If the task name changes, it may be necessary to update the search pattern "HCLTech-Cello-Reach-v0"  
    (in the scripts/list\_envs.py file) so that it can be listed.  
    \# use 'FULL\_PATH\_TO\_isaaclab.sh|bat \-p' instead of 'python' if Isaac Lab is not installed in Python venv or conda  
    python scripts/list\_envs.py

  * Running the task:  
    \# use 'FULL\_PATH\_TO\_isaaclab.sh|bat \-p' instead of 'python' if Isaac Lab is not installed in Python venv or conda  
    python scripts/\<RL\_LIBRARY\>/train.py \--task=HCLTech-Cello-Reach-v0

  * Replaying and exporting the policy:  
    \# use 'FULL\_PATH\_TO\_isaaclab.sh|bat \-p' instead of 'python' if Isaac Lab is not installed in Python venv or conda  
    python scripts/\<RL\_LIBRARY\>/play.py \--task=HCLTech-Cello-Reach-v0

  * Running the task with dummy agents:  
    These include dummy agents that output zero or random actions. They are useful to ensure that the environments are configured correctly.  
    * Zero-action agent  
      \# use 'FULL\_PATH\_TO\_isaaclab.sh|bat \-p' instead of 'python' if Isaac Lab is not installed in Python venv or conda  
      python scripts/zero\_agent.py \--task=HCLTech-Cello-Reach-v0

    * Random-action agent  
      \# use 'FULL\_PATH\_TO\_isaaclab.sh|bat \-p' instead of 'python' if Isaac Lab is not installed in Python venv or conda  
      python scripts/random\_agent.py \--task=HCLTech-Cello-Reach-v0

### **Set up IDE (Optional)**

To setup the IDE, please follow these instructions:

* Run VSCode Tasks, by pressing Ctrl+Shift+P, selecting Tasks: Run Task and running the setup\_python\_env in the drop down menu.  
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the .vscode directory.  
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.  
This helps in indexing all the python modules for intelligent suggestions while writing code.

### **Setup as Omniverse Extension (Optional)**

We provide an example UI extension that will load upon enabling your extension defined in source/hcltech\_cello\_reach/hcltech\_cello\_reach/ui\_extension\_example.py.  
To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:  
   * Navigate to the extension manager using Window \-\> Extensions.  
   * Click on the **Hamburger Icon**, then go to Settings.  
   * In the Extension Search Paths, enter the absolute path to the source directory of this project/repository.  
   * If not already present, in the Extension Search Paths, enter the path that leads to Isaac Lab's extension directory directory (IsaacLab/source)  
   * Click on the **Hamburger Icon**, then click Refresh.  
2. **Search and enable your extension**:  
   * Find your extension under the Third Party category.  
   * Toggle it to enable your extension.

## **Code formatting**

We have a pre-commit template to automatically format your code.  
To install pre-commit:  
pip install pre-commit

Then you can run pre-commit with:  
pre-commit run \--all-files

## **Troubleshooting**

### **Pylance Missing Indexing of Extensions**

In some VsCode versions, the indexing of part of the extensions is missing.  
In this case, add the path to your extension in .vscode/settings.json under the key "python.analysis.extraPaths".  
{  
    "python.analysis.extraPaths": \[  
        "\<path-to-ext-repo\>/source/hcltech\_cello\_reach"  
    \]  
}

### **Pylance Crash**

If you encounter a crash in pylance, it is probable that too many files are indexed and you run out of memory.  
A possible solution is to exclude some of omniverse packages that are not used in your project.  
To do so, modify .vscode/settings.json and comment out packages under the key "python.analysis.extraPaths"  
Some examples of packages that can likely be excluded are:  
"\<path-to-isaac-sim\>/extscache/omni.anim.\*"         // Animation packages  
"\<path-to-isaac-sim\>/extscache/omni.kit.\*"          // Kit UI tools  
"\<path-to-isaac-sim\>/extscache/omni.graph.\*"        // Graph UI tools  
"\<path-to-isaac-sim\>/extscache/omni.services.\*"     // Services tools  
...
