Your **backend** code project. It contains the code project(s) for all the backend subsystems.

1. If the backend is only designed for a web application then this folder can just be used directly for its source project (e.g. a Node.js project)
2. If the backend includes other backend subsystems (e.g. AI model, blockchain smart contract, etc.) then for each of these subsystems, you need to create a subfolder for its code project:
   - **Scenario 1**: web backend + AI model requires these subfolders:
     - `webapp`: for the backend web application development
     - `ai`: for the AI model development
   - **Scenario 2**: web backend + blockchain/smart contract require these subfolders:
     - `webapp`: for the backend web application development
     - `dapp`: (distributed app) for smart contract development