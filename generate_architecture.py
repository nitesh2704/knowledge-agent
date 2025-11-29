"""
Generate a comprehensive architecture diagram for the Knowledge Base Agent using Pillow.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_architecture_diagram():
    """Create a detailed system architecture diagram as PNG."""
    
    # Canvas setup
    width, height = 1400, 1800
    bg_color = (15, 22, 36)  # #0f1624
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        heading_font = ImageFont.truetype("arial.ttf", 16)
        text_font = ImageFont.truetype("arial.ttf", 12)
        small_font = ImageFont.truetype("arial.ttf", 10)
    except:
        title_font = ImageFont.load_default()
        heading_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Colors
    primary = (255, 107, 74)     # #ff6b4a
    secondary = (255, 141, 96)   # #ff8d60
    box_bg = (45, 58, 72)        # #2d3a48
    text_color = (240, 246, 252) # #f0f6fc
    label_color = (152, 177, 201) # #98b1c9
    accent_blue = (77, 168, 255) # #4da8ff
    yellow = (255, 215, 0)       # #ffd700
    
    def draw_box(x, y, w, h, text, bg_color, text_color, multiline=False):
        """Draw a rounded rectangle box with text."""
        radius = 10
        draw.rounded_rectangle([(x, y), (x+w, y+h)], radius=radius, fill=bg_color, outline=text_color, width=2)
        if multiline:
            lines = text.split('\n')
            line_height = 18
            total_height = len(lines) * line_height
            start_y = y + (h - total_height) // 2
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=text_font)
                text_w = bbox[2] - bbox[0]
                draw.text((x + (w - text_w) // 2, start_y + i * line_height), line, fill=text_color, font=text_font)
        else:
            bbox = draw.textbbox((0, 0), text, font=text_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x + (w - text_w) // 2, y + (h - text_h) // 2), text, fill=text_color, font=text_font)
    
    def draw_cylinder(x, y, w, h, text, color):
        """Draw a cylinder shape for database."""
        ellipse_h = 20
        # Top ellipse
        draw.ellipse([(x, y), (x+w, y+ellipse_h)], fill=color, outline=text_color, width=2)
        # Body
        draw.rectangle([(x, y+ellipse_h//2), (x+w, y+h-ellipse_h//2)], fill=color, outline=text_color, width=2)
        # Bottom ellipse
        draw.ellipse([(x, y+h-ellipse_h), (x+w, y+h)], fill=color, outline=text_color, width=2)
        # Text
        lines = text.split('\n')
        line_height = 16
        start_y = y + h//2 - (len(lines) * line_height)//2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=text_font)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (w - text_w) // 2, start_y + i * line_height), line, fill=(255,255,255), font=text_font)
    
    def draw_arrow(x1, y1, x2, y2, label="", color=label_color):
        """Draw an arrow with optional label."""
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        # Arrowhead
        arrow_size = 8
        if x2 > x1:  # Right arrow
            draw.polygon([(x2, y2), (x2-arrow_size, y2-arrow_size//2), (x2-arrow_size, y2+arrow_size//2)], fill=color)
        elif x2 < x1:  # Left arrow
            draw.polygon([(x2, y2), (x2+arrow_size, y2-arrow_size//2), (x2+arrow_size, y2+arrow_size//2)], fill=color)
        elif y2 > y1:  # Down arrow
            draw.polygon([(x2, y2), (x2-arrow_size//2, y2-arrow_size), (x2+arrow_size//2, y2-arrow_size)], fill=color)
        else:  # Up arrow
            draw.polygon([(x2, y2), (x2-arrow_size//2, y2+arrow_size), (x2+arrow_size//2, y2+arrow_size)], fill=color)
        
        # Label
        if label:
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            lines = label.split('\n')
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=small_font)
                text_w = bbox[2] - bbox[0]
                draw.text((mid_x - text_w // 2, mid_y - 20 + i*12), line, fill=color, font=small_font)
    
    # Title
    title_text = "Knowledge Base Agent - RAG System Architecture"
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text((width//2 - title_w//2, 30), title_text, fill=primary, font=title_font)
    
    # Layer 1: UI (top)
    y_offset = 100
    draw.text((50, y_offset), "User Interface Layer", fill=secondary, font=heading_font)
    y_offset += 35
    draw_box(100, y_offset, 250, 60, "Streamlit Web UI\n(app.py)", box_bg, text_color, True)
    draw_box(400, y_offset, 200, 60, "File Upload\nPDF/DOCX/TXT", box_bg, text_color, True)
    draw_box(650, y_offset, 180, 60, "Question Input", box_bg, text_color, True)
    draw_box(880, y_offset, 200, 60, "Answer Display\n+ Sources", box_bg, text_color, True)
    
    # Layer 2: Core
    y_offset = 240
    draw.text((50, y_offset), "Application Core", fill=secondary, font=heading_font)
    y_offset += 35
    draw_box(80, y_offset, 200, 60, "Document\nIngestion", box_bg, text_color, True)
    draw_box(320, y_offset, 180, 60, "Text Chunking\nsize=1200", box_bg, text_color, True)
    draw_box(540, y_offset, 200, 60, "Query\nProcessing", box_bg, text_color, True)
    draw_box(780, y_offset, 200, 60, "Semantic\nRetrieval (Top-5)", box_bg, text_color, True)
    draw_box(1020, y_offset, 180, 60, "Prompt\nEngineering", box_bg, text_color, True)
    
    # Layer 3: Utils
    y_offset = 400
    draw.text((50, y_offset), "Utility Functions (utils.py)", fill=secondary, font=heading_font)
    y_offset += 35
    draw_box(150, y_offset, 220, 60, "Document Parsers\npdfplumber, docx", box_bg, text_color, True)
    draw_box(420, y_offset, 200, 60, "gemini_embed()\nParallel Workers", box_bg, text_color, True)
    draw_box(670, y_offset, 200, 60, "gemini_chat()\nAnswer Gen", box_bg, text_color, True)
    
    # Layer 4: External APIs
    y_offset = 560
    draw.text((50, y_offset), "External Services (Google Gemini API)", fill=secondary, font=heading_font)
    y_offset += 35
    draw_box(200, y_offset, 240, 70, "text-embedding-004\n768-dim Vectors\nRetry + Backoff", primary, (255,255,255), True)
    draw_box(500, y_offset, 240, 70, "gemini-2.0-flash\nChat Completion\nContext-Aware", primary, (255,255,255), True)
    
    # Layer 5: Storage
    y_offset = 720
    draw.text((50, y_offset), "Persistent Storage", fill=secondary, font=heading_font)
    y_offset += 35
    draw_cylinder(250, y_offset, 200, 80, "ChromaDB\nVector Database\nCosine Similarity", accent_blue)
    draw_cylinder(550, y_offset, 200, 80, "Metadata Store\nChunk IDs\nSources", accent_blue)
    
    # Config box
    draw_box(900, 720, 280, 80, ".env Configuration\nAPI Keys\nCHUNK_SIZE=1200\nEMBED_WORKERS=4", (100, 100, 50), yellow, True)
    
    # Flow section
    y_offset = 900
    draw.text((50, y_offset), "Data Flows", fill=secondary, font=heading_font)
    
    # Ingestion flow (left side)
    y_offset += 35
    draw.text((80, y_offset), "Ingestion Flow:", fill=text_color, font=text_font)
    flow_x = 80
    flow_y = y_offset + 30
    steps = [
        "1. Upload Document",
        "2. Parse (PDF/DOCX/TXT)",
        "3. Split into Chunks",
        "4. Parallel Embedding",
        "5. Store Vectors in ChromaDB"
    ]
    for i, step in enumerate(steps):
        draw.text((flow_x, flow_y + i*30), f"→ {step}", fill=label_color, font=text_font)
    
    # Query flow (right side)
    flow_x = 700
    draw.text((flow_x, y_offset), "Query Flow:", fill=text_color, font=text_font)
    flow_y = y_offset + 30
    steps_query = [
        "1. User asks question",
        "2. Embed query with Gemini",
        "3. Vector search in ChromaDB",
        "4. Retrieve Top-5 chunks",
        "5. Build prompt with context",
        "6. Generate answer with LLM",
        "7. Display answer + sources"
    ]
    for i, step in enumerate(steps_query):
        draw.text((flow_x, flow_y + i*30), f"→ {step}", fill=label_color, font=text_font)
    
    # Key features box
    y_offset = 1150
    draw.text((50, y_offset), "Key Features & Optimizations", fill=secondary, font=heading_font)
    y_offset += 35
    features = [
        "✓ Multi-format support: PDF, DOCX, TXT parsing",
        "✓ Parallel embedding generation (4 workers default)",
        "✓ Optimized chunking: 1200 chars, 80 overlap",
        "✓ Persistent vector storage with ChromaDB",
        "✓ Semantic retrieval with cosine similarity",
        "✓ Context-aware answer generation",
        "✓ Source attribution for transparency",
        "✓ Retry logic with exponential backoff",
        "✓ Clean UI with Streamlit custom styling"
    ]
    for i, feature in enumerate(features):
        draw.text((80, y_offset + i*28), feature, fill=text_color, font=text_font)
    
    # Technology stack box
    y_offset = 1500
    draw.text((50, y_offset), "Technology Stack", fill=secondary, font=heading_font)
    y_offset += 35
    tech_items = [
        ("Frontend:", "Streamlit 1.51.0"),
        ("Vector DB:", "ChromaDB 1.3.5 (cosine distance)"),
        ("Embeddings:", "Google Gemini text-embedding-004"),
        ("LLM:", "Google Gemini 2.0 Flash"),
        ("Parsing:", "pdfplumber, python-docx"),
        ("Language:", "Python 3.10+")
    ]
    for i, (label, value) in enumerate(tech_items):
        draw.text((80, y_offset + i*28), f"{label} {value}", fill=text_color, font=text_font)
    
    # Performance notes
    y_offset = 1680
    draw.text((50, y_offset), "Performance Characteristics", fill=secondary, font=heading_font)
    y_offset += 35
    perf_notes = [
        "• Parallel embedding: ~4x faster than sequential",
        "• Larger chunks (1200 vs 800): fewer API calls, ~30% speed up",
        "• Vector search: sub-second retrieval on 1000s of chunks",
        "• End-to-end query latency: ~3-5 seconds typical"
    ]
    for i, note in enumerate(perf_notes):
        draw.text((80, y_offset + i*25), note, fill=label_color, font=small_font)
    
    # Save
    output_path = "architecture_diagram.png"
    img.save(output_path)
    print(f"✓ Architecture diagram saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_architecture_diagram()
