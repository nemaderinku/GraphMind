# 🤖 Enhanced Multi-Agent CSV Processor

An intelligent FastAPI application that uses multiple AI agents to collaboratively clean, analyze, and visualize CSV data in real-time. Watch AI agents work together to transform your data into stunning D3.js visualizations!

## ✨ Features

- **🔧 Agent A**: Data cleaning and standardization
- **📊 Agent B**: Data analysis and visualization recommendations  
- **🎨 Agent C**: D3.js chart code generation
- **🔍 Agent D**: Code review and optimization
- **📡 Real-time streaming**: Watch agents collaborate live
- **🚀 Dual chart creation**: Generate multiple complementary visualizations
- **💬 Live collaboration**: See Agent C ↔ D conversations in real-time

## 🛠️ Setup Instructions

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd multi-agent-csv-processor
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_BASE_URL=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=o3-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### 5. Update main.py for Environment Variables

Replace the imports section in `main.py`:

```python
# Replace this line:
# from keys_sensitive import API_KEY, BASE_URL

# With this:
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL")
MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-mini")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
```

### 6. Install python-dotenv
```bash
pip install python-dotenv
```

### 7. Start the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🚀 Usage

### Web Interface
1. Open your browser to `http://localhost:8000`
2. Upload a CSV file or paste CSV text
3. Click "🚀 Process with AI Agents (Real-time)"
4. Watch the agents collaborate in real-time!
5. View your charts in a new tab when processing completes

### API Endpoints

#### Health Check
```bash
GET /health
```
Check if the service and Azure OpenAI connection are working.

#### Start Processing (Streaming)
```bash
POST /start_processing
```
**Parameters:**
- `file`: CSV file upload
- `csv_text`: Raw CSV text (Form data)

**Returns:** Session ID for streaming

#### Stream Updates
```bash
GET /stream/{session_id}
```
Server-Sent Events stream with real-time updates:
- `agent_start`: Agent begins working
- `agent_complete`: Agent finishes task
- `agent_interaction`: Live C↔D collaboration
- `processing_complete`: Final results with chart code

#### Legacy Processing (Non-streaming)
```bash
POST /process_csv
```
Traditional endpoint that returns complete results at once.

## 📊 Sample Data

The interface includes sample employee data:
```csv
Name,Age,City,Salary
John,25,New York,50000
Jane,30,Los Angeles,65000
Bob,35,Chicago,55000
Alice,28,Seattle,58000
Mike,32,Boston,62000
```

## 🏗️ Architecture

### Agent Workflow
1. **Agent A** cleans and standardizes the CSV data
2. **Agent B** analyzes patterns and suggests 2 different chart types
3. **Agent C** creates D3.js visualization code with embedded data
4. **Agent D** reviews and provides feedback for improvements
5. **Collaborative Loop**: C & D iterate until charts are perfected

### Technology Stack
- **Backend**: FastAPI, Python 3.9+
- **AI**: Azure OpenAI (o3-mini model)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Visualizations**: D3.js v7
- **Streaming**: Server-Sent Events (SSE)

## 🔧 Troubleshooting

### Common Issues

**1. "Please update AZURE_ENDPOINT" error**
- Ensure your `.env` file has the correct `AZURE_OPENAI_BASE_URL`
- Format: `https://your-resource-name.openai.azure.com`

**2. "Please update API_KEY" error**  
- Check your `AZURE_OPENAI_API_KEY` in the `.env` file
- Ensure no extra spaces or quotes

**3. Connection errors**
- Verify Azure OpenAI resource is active
- Check API key permissions
- Confirm deployment name matches your Azure setup

**4. Charts not opening**
- Enable pop-ups in your browser
- Check browser console for JavaScript errors

### Health Check
Visit `http://localhost:8000/health` to verify:
- ✅ Service is running
- ✅ Azure OpenAI connection is working
- ✅ Model deployment is accessible

## 🎯 Key Features Explained

### Real-time Streaming
- Uses Server-Sent Events for live updates
- See agent progress as it happens
- Watch C & D collaborate with live conversation

### Dual Chart Generation  
- Agent B suggests 2 complementary chart types
- Agent C creates both visualizations in one HTML file
- Charts include hover tooltips and smooth animations

### Intelligent Collaboration
- Agent D reviews Agent C's code
- Provides specific feedback for improvements
- Iterative refinement until charts are perfect

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

- Check `/health` endpoint for diagnostics
- Review browser console for frontend issues
- Check server logs for backend problems