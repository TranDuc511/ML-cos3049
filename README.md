# Underground Transaction Detection

A full-stack application to detect fraudulent transactions using machine learning.

## 📂 Project Structure

```text
.
├── backend/       # Contains the core server logic, application APIs, and all machine learning operations.
│   ├── ai/        # Houses the machine learning dataset, training scripts, and saved AI models.
│   └── webapp/    # Connects the AI models to the internet via an accessible API service built with FastAPI.
├── frontend/      # Manages everything the user sees and interacts with directly on their screen.
│   └── web/       # The React web application where the visual dashboards and form configurations exist.
├── .gitignore     # Specifies which files and directories should be excluded from version control.
└── README.md      # Serves as the main entry-point documentation explaining how to navigate and launch the project.
```

## 🚀 How to Run the Project

Below are the complete steps to run the application on your computer. You do not need any advanced coding skills to start the project; just follow these instructions exactly.

To run the whole application, you will need to open **two** separate command terminal windows (such as Command Prompt, Terminal, or PowerShell) located in this project's root folder.

### Step 1: Start the Backend (The Data Server)

This step initializes the system that performs the fraud detection logic.

1. Open a new terminal.
2. Install the required tools by typing these commands one by one, pressing Enter after each:

   ```bash
   cd backend
   pip install -r ai/ML/requirements.txt
   pip install -r webapp/requirements.txt
   ```

3. Start the server application:

   ```bash
   cd webapp
   uvicorn main:app --reload
   ```

4. **Keep this terminal window open.** The backend is now successfully running at `http://localhost:8000`.

### Step 2: Start the Frontend (The User Interface)

This step launches the visual website that you will actually click on and use.

1. Open a **second, completely new terminal** at the project root.
2. Run the following commands one by one to install the website packages and start the visual application:

   ```bash
   cd frontend/web
   npm install
   npm run dev
   ```

3. **Keep this terminal window open as well.** The frontend is now live!

### Step 3: View the Dashboard

Open your preferred web browser (e.g., Chrome, Edge, Safari) and go to:
<http://localhost:5173>

