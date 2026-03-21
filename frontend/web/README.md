# Underground Transaction Detection (React Version)

A modern, React-based dashboard for detecting and analyzing underground or suspicious financial transactions. This version enhances the original project with a component-based architecture using Vite and React.

## Overview

This project provides a comprehensive interface for financial analysts to monitor transactions, segment customers based on behavioral patterns, and generate AI-driven reports on potential risks.

## Features

- **Component-Based Architecture**: Highly modular React components for better maintainability.
- **Dynamic View Switching**: Smooth navigation between dashboard views using React state.
- **Deep Search & Filtering**: Efficient data filtering for Transactions and Customers using `useMemo`.
- **Advanced File Upload**: Interactive drag-and-drop interface for dataset ingestion.
- **Modern Styling**: Responsive design powered by Vanilla CSS and CSS Variables.

## Tech Stack

- **React 18**: Frontend library for building the user interface.
- **Vite**: Ultra-fast build tool and development server.
- **Vanilla CSS**: Custom professional design system.
- **React Router Dom**: (Installed for future routing enhancements).

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (Project initialized with npm)

### Installation

1.  Clone or navigate to the project directory.
2.  Install dependencies:
    ```bash
    npm install
    ```

### Running the Application

1.  Start the development server:
    ```bash
    npm run dev
    ```
2.  Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:5173`).

## Project Structure

- `src/components/`: Reusable UI components (Sidebar, etc.)
- `src/views/`: Main page-level components (Overview, Transactions, etc.)
- `src/App.jsx`: Root component handling authentication and layout.
- `src/index.css`: Global styles and design system tokens.
- `legacy/`: Contains the original HTML/JS files for reference.

## License

MIT
