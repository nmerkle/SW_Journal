# SW_Journal
Here you can find the source code and generated datasets regarding the domestic activities that were generated in the scope of the submitted Semantic Web Journal paper. The generated data is based on the **[Virtual Home](https://github.com/xavierpuigf/virtualhome)** dataset, which is published under the MIT license.

### Directory Structure
* **code** -- Contains the source code
* **Entity_Embeddings** -- contains the trained entity embeddings
* **VirtualHomeKG.zip** -- contains the generated semantic task descriptions of all activities given in the Virtual Home dataset.

### Preparations
The demo was tested on a Windows OS.

* Install **[Nodejs + NPM](https://nodejs.org/en/download/)** on your host system.
* Clone **this repo** to your host system.

---

__Configuration of the Demo (optional):__

It is not required to change the configuration of the demo. However, you have the possibility to specify the number of tasks that shall be evaluated as well as the radius hyper-parameter of the executed algorithm. The **[.env](code/.env)** configuration file contains all configurable parameters as key-value pairs. The following parameters can be defined:
* NUMBER_TASKS -- The number of tasks that shall be evaluated. Default value is set to 52 as performed in the evaluation of the presented approach. 
* RADIUS -- The search radius for finding nearby embedding vectors. Default value is set to 0.25

---

__Start Demo:__ 
* Go into the **code** directory and execute via command line: 
``` console
npm install
```
All required javascript libraries are installed automatically.

* Finally, execute:
``` console
npm start 
```

* (Optional) The deep neural network for training the entity embeddings can be executed via:
``` console
node ./EmbeddingTrainer.js
```

__Obtained Results:__
 
Two files (AgentEnsembleEval.csv and ComposedPolicies.json) will be generated that contain the evaluation results, such as task name, required steps, number of episodes, cumulative reward, wrong decisions and the policies that were composed for each evaluated task. Furthermore, a directory named **../VirtualHomeKG** will be generated that contains two directories (**JSONLD** and **TURTLE**). In both directories the semantic task representations for the simulation function are stored in JSONLD and Turtle format.


