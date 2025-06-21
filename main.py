#!/usr/bin/env python3
"""
Enhanced Multi-Agent CSV Processing FastAPI Application with Streaming
"""

import io
import os
import asyncio
import uuid
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import pandas as pd
import time
import json
from typing import List, Dict, Generator
import constants
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_BASE_URL")
MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# In-memory storage for active processing sessions
active_sessions = {}

# Create FastAPI app
app = FastAPI(title="Multi-Agent CSV Processor", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentPrompts:
    """Enhanced agent prompt templates"""

    A_CONTENT = """You are Agent A, a data cleaning specialist. Your job is to:
1. Clean and standardize CSV data
2. Fix formatting issues, remove duplicates
3. Handle missing values appropriately
4. Ensure consistent data types
5. Return the cleaned data in CSV format

IMPORTANT: Always explain what operations you performed. Start your response with a summary like:
"CLEANING OPERATIONS PERFORMED:
- Removed X duplicate rows
- Fixed Y inconsistent data formats
- Handled Z missing values
- Standardized column names

CLEANED DATA:"

Then provide the cleaned CSV data."""

    B_CONTENT = """You are Agent B, a data visualization expert.
Your job is to suggest 2 DIFFERENT and complementary D3.js-compatible visualizations for the given data.

Chart types you can use: "bar", "line", "scatter", "pie", "area", "donut", "histogram", "heatmap"

Analyze the data and suggest TWO different charts with these exact fields for each:
- title: Clear, descriptive title
- chart_type: Choose from the types above - make them DIFFERENT types
- x_axis: Field name for X-axis (must exist in data)
- y_axis: Field name for Y-axis (must exist in data) 
- grouping: Field for grouping/color coding (optional)
- color: Hex color code (e.g., "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
- x_label: Human-readable X-axis label
- y_label: Human-readable Y-axis label

Return ONLY a JSON array with exactly 2 chart objects. Use actual field names from the data.
Make the charts show DIFFERENT perspectives - avoid bar+scatter, use variety like pie+line, area+donut, etc."""

    C_CONTENT = """You are Agent C, a frontend engineer. You create D3.js charts that wow users.
You will receive 2 chart specifications and data. Create HTML with TWO charts:

CRITICAL REQUIREMENTS:
- Create complete standalone HTML document with <html>, <head>, <body>
- Include D3.js v7 CDN: <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
- EMBED THE ACTUAL DATA directly in the JavaScript code as a variable (do NOT try to load external data)
- Use <div id="chart1"> and <div id="chart2"> containers side by side
- Create working visualizations for both charts using the specs provided
- Use exact colors and labels from specs
- ADD SMOOTH HOVER TOOLTIPS to both charts showing data values
- Add smooth animations (transitions) for loading and interactions
- Include chart titles above each visualization
- Make charts responsive with proper sizing (each chart ~500px wide)
- Add CSS styling for professional layout

IMPORTANT - DATA EMBEDDING:
Parse the CSV data provided and embed it directly in your JavaScript like this:
const data = [
  {name: "John", age: 25, salary: 50000},
  {name: "Jane", age: 30, salary: 65000},
  // ... all data rows
];

DO NOT use d3.csv() or any external data loading. The HTML must be completely self-contained.

Example tooltip code:
.on("mouseover", function(event, d) {
    tooltip.style("opacity", 1)
        .html("Value: " + d.value)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 10) + "px");
})
.on("mouseout", function() {
    tooltip.style("opacity", 0);
});

Return complete, working HTML that renders both charts beautifully with tooltips and embedded data."""

    C_FIXER_CONTENT = constants.AGENT_C_FIXER_PROMPT

    D_CONTENT = constants.AGENT_D_SYSTEM_PROMPT


def query_azure_openai(messages: List[Dict], max_retries: int = 3) -> str:
    """Query Azure OpenAI with proper error handling and retries"""

    if AZURE_ENDPOINT == "https://YOUR-RESOURCE-NAME.openai.azure.com":
        raise Exception("Please update AZURE_ENDPOINT in the configuration")
    if API_KEY == "YOUR-ACTUAL-API-KEY":
        raise Exception("Please update API_KEY in the configuration")

    url = f"{AZURE_ENDPOINT}/openai/deployments/{MODEL_DEPLOYMENT}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    params = {"api-version": API_VERSION}

    payload = {
        "messages": messages,
        "max_completion_tokens": 4000,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                params=params,
                timeout=60
            )

            if response.status_code == 200:
                res_json = response.json()
                if 'choices' not in res_json:
                    raise Exception(f"API error: {res_json}")
                return res_json['choices'][0]['message']['content']
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                if attempt == max_retries - 1:
                    raise Exception(error_msg)
                time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Request failed: {str(e)}")
            time.sleep(2 ** attempt)

    raise Exception("All retry attempts failed")


def send_sse_message(event_type: str, data: dict) -> str:
    """Format data as Server-Sent Event (kept for compatibility)"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
async def root():
    return {"message": "Enhanced Multi-Agent CSV Processor API", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    try:
        test_messages = [{"role": "user", "content": "Hello, respond with 'OK' if you're working."}]
        response = query_azure_openai(test_messages)
        return {
            "status": "healthy",
            "azure_openai": "connected",
            "model": MODEL_DEPLOYMENT,
            "endpoint": AZURE_ENDPOINT,
            "test_response": response
        }
    except Exception as e:
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e),
                "azure_openai": "disconnected"
            },
            status_code=500
        )


@app.post("/start_processing")
async def start_processing(file: UploadFile = None, csv_text: str = Form(None)):
    """Start CSV processing and return a session ID"""

    try:
        # Read input CSV
        if file:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
            csv_data = df.to_csv(index=False)
        elif csv_text:
            df = pd.read_csv(io.StringIO(csv_text))
            csv_data = csv_text
        else:
            return JSONResponse({"error": "No CSV provided"}, status_code=400)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Initialize session data
        active_sessions[session_id] = {
            "status": "starting",
            "csv_data": csv_data,
            "events": [],
            "completed": False,
            "error": None
        }

        # Start background processing
        asyncio.create_task(process_csv_background(session_id, csv_data))

        return {"session_id": session_id, "status": "started"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/stream/{session_id}")
async def stream_processing_updates(session_id: str):
    """Stream real-time updates for a processing session"""

    if session_id not in active_sessions:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    async def event_generator():
        try:
            session = active_sessions[session_id]
            sent_events = 0

            while not session["completed"] and session["error"] is None:
                # Send any new events
                while sent_events < len(session["events"]):
                    event = session["events"][sent_events]
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                    sent_events += 1

                # Wait a bit before checking again (reduced from 0.1 to 0.05 for faster updates)
                await asyncio.sleep(0.05)

            # Send final events
            while sent_events < len(session["events"]):
                event = session["events"][sent_events]
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                sent_events += 1

            # Send completion or error
            if session["error"]:
                yield f"event: error\ndata: {json.dumps({'message': session['error']})}\n\n"

            # Clean up session after delay
            asyncio.create_task(cleanup_session(session_id))

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


async def cleanup_session(session_id: str):
    """Clean up session after delay"""
    await asyncio.sleep(30)  # Keep session for 30 seconds after completion
    if session_id in active_sessions:
        del active_sessions[session_id]


async def process_csv_background(session_id: str, csv_data: str):
    """Background task to process CSV and emit events"""

    try:
        session = active_sessions[session_id]

        def emit_event(event_type: str, data: dict):
            session["events"].append({"type": event_type, "data": data})
            # Simplified logging without N/A values
            agent = data.get('agent', '')
            message = data.get('message', '')
            action = data.get('action', '')

            log_msg = f"🔔 Event: {event_type}"
            if agent:
                log_msg += f" | Agent: {agent}"
            if action:
                log_msg += f" | Action: {action[:50]}..."
            elif message:
                log_msg += f" | Message: {message[:50]}..."

            print(log_msg)

        # Initialize timing and conversation log
        start_time = time.time()
        agent_c_d_log = []

        print(f"🚀 Starting background processing for session: {session_id}")

        # Send initial status
        emit_event("status", {
            "message": "Starting multi-agent processing...",
            "current_agent": "setup"
        })

        # Agent A: Data Cleaning
        print("🔧 Starting Agent A (Data Cleaning)...")
        emit_event("agent_start", {
            "agent": "A",
            "task": "Data Cleaning",
            "status": "processing"
        })

        # Add small delay to ensure event is processed
        await asyncio.sleep(0.1)

        t0 = time.time()
        agent_a_prompt = [
            {"role": "system", "content": AgentPrompts.A_CONTENT},
            {"role": "user", "content": f"Clean this CSV and explain what operations you performed:\n{csv_data}"}
        ]

        cleaned_data = query_azure_openai(agent_a_prompt)
        t1 = time.time()

        print(f"✅ Agent A completed in {t1 - t0:.2f}s")
        emit_event("agent_complete", {
            "agent": "A",
            "task": "Data Cleaning",
            "result": cleaned_data,
            "duration": round(t1 - t0, 2),
            "status": "completed"
        })

        # Add small delay to ensure event is processed
        await asyncio.sleep(0.1)

        # Agent B: Data Insights
        print("📊 Starting Agent B (Data Insights)...")
        emit_event("agent_start", {
            "agent": "B",
            "task": "Data Analysis",
            "status": "processing"
        })

        # Add small delay to ensure event is processed
        await asyncio.sleep(0.1)

        agent_b_prompt = [
            {"role": "system", "content": AgentPrompts.B_CONTENT},
            {"role": "user",
             "content": f"Analyze this cleaned data and suggest 2 DIFFERENT types of complementary visualizations (avoid repetitive bar+scatter, use variety like pie+line, area+donut, etc.):\n\n{cleaned_data}"}
        ]

        graph_ideas = query_azure_openai(agent_b_prompt)
        t2 = time.time()

        print(f"✅ Agent B completed in {t2 - t1:.2f}s")
        emit_event("agent_complete", {
            "agent": "B",
            "task": "Data Analysis",
            "result": graph_ideas,
            "duration": round(t2 - t1, 2),
            "status": "completed"
        })

        # Add small delay to ensure event is processed
        await asyncio.sleep(0.1)

        # Agent C & D Collaborative Process
        emit_event("collaboration_start", {
            "message": "Starting Agent C & D collaboration...",
            "agents": ["C", "D"]
        })

        # Agent C: Initial Chart Creation
        emit_event("agent_action", {
            "agent": "C",
            "action": "Creating initial chart code...",
            "status": "processing"
        })

        agent_c_prompt = [
            {"role": "system", "content": AgentPrompts.C_CONTENT},
            {"role": "user",
             "content": f"Create 2 D3.js charts with HOVER TOOLTIPS and EMBEDDED DATA using these specs and data:\n\nChart Specifications:\n{graph_ideas}\n\nCSV Data to embed directly in JavaScript:\n{cleaned_data}\n\nIMPORTANT: Parse this CSV data and embed it directly as a JavaScript array in your HTML. Do NOT use d3.csv() or try to load external data. The HTML must be completely standalone and self-contained with all data embedded."}
        ]

        chart_code = query_azure_openai(agent_c_prompt).strip()
        t3 = time.time()

        print(f"✅ Agent C completed in {t3 - t2:.2f}s")
        print(f"📊 Chart code length: {len(chart_code)} characters")
        print(f"📊 Chart code preview: {chart_code[:200]}...")

        # Log and emit Agent C's initial work
        c_entry = {
            "role": "Agent C",
            "action": "Initial dual chart creation",
            "content": chart_code,
            "timestamp": time.time(),
            "duration": round(t3 - t2, 2)
        }
        agent_c_d_log.append(c_entry)

        emit_event("agent_interaction", {
            "interaction": c_entry,
            "conversation_length": len(agent_c_d_log)
        })

        # Add small delay for visual effect
        await asyncio.sleep(0.2)
        max_iterations = 3
        current_code = chart_code

        for iteration in range(max_iterations):
            # Agent D: Review the code
            emit_event("agent_action", {
                "agent": "D",
                "action": f"Reviewing code (iteration {iteration + 1})...",
                "status": "processing"
            })

            agent_d_prompt = [
                {"role": "system", "content": AgentPrompts.D_CONTENT},
                {"role": "user", "content": f"Review this D3.js code:\n\n{current_code}"}
            ]

            d_review = query_azure_openai(agent_d_prompt).strip()
            t_d = time.time()

            # Log Agent D's review
            d_entry = {
                "role": "Agent D",
                "action": f"Code review (iteration {iteration + 1})",
                "content": d_review,
                "timestamp": time.time(),
                "iteration": iteration + 1
            }
            agent_c_d_log.append(d_entry)

            emit_event("agent_interaction", {
                "interaction": d_entry,
                "conversation_length": len(agent_c_d_log)
            })

            # Check if Agent D approved the code
            if "✅ Valid ✅" in d_review or "valid" in d_review.lower():
                emit_event("collaboration_complete", {
                    "message": f"Agent D approved the code after {iteration + 1} iteration(s)!",
                    "final_code": current_code,
                    "iterations": iteration + 1
                })
                break

            # If not approved and not the last iteration, Agent C fixes the code
            if iteration < max_iterations - 1:
                emit_event("agent_action", {
                    "agent": "C",
                    "action": f"Fixing code based on feedback (iteration {iteration + 1})...",
                    "status": "processing"
                })

                agent_c_fix_prompt = [
                    {"role": "system", "content": AgentPrompts.C_FIXER_CONTENT},
                    {"role": "user",
                     "content": f"Fix this code based on the feedback:\n\nFeedback:\n{d_review}\n\nCurrent Code:\n{current_code}"}
                ]

                fixed_code = query_azure_openai(agent_c_fix_prompt).strip()
                current_code = fixed_code
                t_c_fix = time.time()

                # Log Agent C's fix
                c_fix_entry = {
                    "role": "Agent C",
                    "action": f"Code fix (iteration {iteration + 1})",
                    "content": fixed_code,
                    "timestamp": time.time(),
                    "iteration": iteration + 1
                }
                agent_c_d_log.append(c_fix_entry)

                emit_event("agent_interaction", {
                    "interaction": c_fix_entry,
                    "conversation_length": len(agent_c_d_log)
                })

        # Final results
        total_time = time.time() - start_time

        emit_event("processing_complete", {
            "cleaned_data": cleaned_data,
            "graph_ideas": graph_ideas,
            "final_chart_code": current_code,
            "agent_c_d_log": agent_c_d_log,
            "processing_time": {
                "agent_a": round(t1 - t0, 2),
                "agent_b": round(t2 - t1, 2),
                "agent_c_d": round(time.time() - t2, 2),
                "total": round(total_time, 2)
            },
            "total_interactions": len(agent_c_d_log),
            "charts_created": 2
        })

        session["completed"] = True

    except Exception as e:
        session["error"] = str(e)
        session["completed"] = True


# Keep the original endpoint for backwards compatibility
@app.post("/process_csv")
async def process_csv(file: UploadFile = None, csv_text: str = Form(None)):
    """Original endpoint - still available for backwards compatibility"""
    try:
        # Read input CSV
        if file:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
            original_csv = df.to_csv(index=False)
        elif csv_text:
            df = pd.read_csv(io.StringIO(csv_text))
            original_csv = csv_text
        else:
            return JSONResponse({"error": "No CSV provided"}, status_code=400)

        print(f"📊 Processing CSV with {len(df)} rows and {len(df.columns)} columns")

        # Initialize conversation log
        agent_c_d_log = []

        # Agent A: Cleaning
        print("🔧 Starting Agent A (Data Cleaning)...")
        t0 = time.time()
        agent_a_prompt = [
            {"role": "system", "content": AgentPrompts.A_CONTENT},
            {"role": "user", "content": f"Clean this CSV:\n{original_csv}"}
        ]

        cleaned_data = query_azure_openai(agent_a_prompt)
        t1 = time.time()
        print(f"✅ Agent A completed in {t1 - t0:.2f}s")

        # Agent B: Insights
        print("📊 Starting Agent B (Data Insights)...")
        agent_b_prompt = [
            {"role": "system", "content": AgentPrompts.B_CONTENT},
            {"role": "user", "content": f"Analyze this cleaned data and suggest visualizations:\n\n{cleaned_data}"}
        ]

        graph_ideas = query_azure_openai(agent_b_prompt)
        t2 = time.time()
        print(f"✅ Agent B completed in {t2 - t1:.2f}s")

        # Agent C: Initial Chart Creation
        print("🎨 Starting Agent C (Chart Creation)...")
        agent_c_prompt = [
            {"role": "system", "content": AgentPrompts.C_CONTENT},
            {"role": "user",
             "content": f"Create a D3.js visualization based on these specs:\n\nSpecs:\n{graph_ideas}\n\nData:\n{cleaned_data}"}
        ]

        chart_code = query_azure_openai(agent_c_prompt).strip()
        t3 = time.time()
        print(f"✅ Agent C completed in {t3 - t2:.2f}s")

        # Log Agent C's work
        agent_c_d_log.append({
            "role": "Agent C (Initial Code)",
            "content": chart_code,
            "full_content": chart_code,
            "timestamp": time.time()
        })

        # Agent D: Code Review and Correction
        print("🔍 Starting Agent D (Code Review)...")
        agent_d_prompt = [
            {"role": "system", "content": AgentPrompts.D_CONTENT},
            {"role": "user",
             "content": f"Review and correct this D3.js chart code:\n\n{chart_code}\n\nEnsure it works with this data:\n{cleaned_data}"}
        ]

        corrected_code = query_azure_openai(agent_d_prompt)
        t4 = time.time()
        print(f"✅ Agent D completed in {t4 - t3:.2f}s")

        # Log Agent D's work
        agent_c_d_log.append({
            "role": "Agent D (Corrected Code)",
            "content": corrected_code,
            "full_content": corrected_code,
            "timestamp": time.time()
        })

        total_time = t4 - t0
        print(f"🎉 All agents completed in {total_time:.2f}s total")

        return {
            "cleaned_data": cleaned_data,
            "graph_ideas": graph_ideas,
            "chart_code": corrected_code,
            "agent_c_d_log": agent_c_d_log,
            "processing_time": {
                "agent_a": round(t1 - t0, 2),
                "agent_b": round(t2 - t1, 2),
                "agent_c": round(t3 - t2, 2),
                "agent_d": round(t4 - t3, 2),
                "total": round(total_time, 2)
            }
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(
            {"error": f"Processing failed: {str(e)}"},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn

    if AZURE_ENDPOINT == "https://YOUR-RESOURCE-NAME.openai.azure.com":
        print("❌ Please update AZURE_ENDPOINT with your actual endpoint")
        exit(1)

    if API_KEY == "YOUR-ACTUAL-API-KEY":
        print("❌ Please update API_KEY with your actual API key")
        exit(1)

    print("🚀 Starting Enhanced Multi-Agent CSV Processor with Streaming...")
    print(f"🔗 Azure Endpoint: {AZURE_ENDPOINT}")
    print(f"🤖 Model Deployment: {MODEL_DEPLOYMENT}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📊 Health check: http://localhost:8000/health")
    print("✨ New features: Real-time streaming, Agent C&D collaboration")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)