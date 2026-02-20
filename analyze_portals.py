import re
import os
import ast

def analyze_html_files(root_dir):
    html_files = []
    exclude_dirs = {'.git', 'node_modules', 'venv', '.venv', '__pycache__', 'data'}
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    
    print(f"Found {len(html_files)} HTML files.")
    
    html_analysis = {}

    for file_path in html_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        fetches = re.findall(r'fetch\([\'"`]((?!http)[^)\'"`]+)[\'"`]', content)
        # Also catch fetch with template literals like `${backend_url}/path`
        template_fetches = re.findall(r'fetch\(`\$\{backend_url\}(/[^`]+)`', content)
        
        all_fetches = fetches + template_fetches
        
        html_analysis[file_path] = {
            'fetches': all_fetches,
            'has_backend_url': 'backend_url' in content
        }
        
    return html_analysis

def analyze_python_backends(root_dir):
    backend_files = [
        os.path.join(root_dir, 'src', 'ui', 'conversational_backend_server.py'),
        # Add others if found, e.g., diegetic_backend.py if it uses Flask/http.server
    ]
    
    backend_analysis = {}
    
    for file_path in backend_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # extract flask routes
        routes = re.findall(r'@app\.route\([\'"]([^)\'"]+)[\'"]', content)
        
        # Check for training logic imports
        has_trainer = 'Trainer' in content or 'backward' in content or 'step' in content
        
        backend_analysis[file_path] = {
            'routes': routes,
            'has_training_logic': has_trainer
        }
        
    return backend_analysis

def main():
    root_dir = os.getcwd()
    with open('portal_analysis_report.txt', 'w', encoding='utf-8') as report_file:
        def log(msg):
            report_file.write(msg + '\n')
            print(msg) # Still print to verify

        log(f"Analyzing portals in {root_dir}...")
        
        html_data = analyze_html_files(root_dir)
        backend_data = analyze_python_backends(root_dir)
        
        log("\n--- HTML Portal Analysis ---")
        for file, data in html_data.items():
            log(f"\nFile: {os.path.basename(file)}")
            if not data['fetches']:
                log("  [WARNING] No API calls found. Likely a mock-up.")
            else:
                for endpoint in data['fetches']:
                    log(f"  - Calls: {endpoint}")
                    
        log("\n--- Backend Analysis ---")
        known_routes = set()
        for file, data in backend_data.items():
            log(f"\nFile: {os.path.basename(file)}")
            log(f"  Training Logic Detected: {data['has_training_logic']}")
            log(f"  Routes: {data['routes']}")
            for r in data['routes']:
                known_routes.add(r)

        log("\n--- Connectivity Check ---")
        
        for file, data in html_data.items():
            for endpoint in data['fetches']:
                # Handle variable interpolation roughly
                clean_endpoint = endpoint.replace('/api/', '/') # heuristic
                
                # Direct match check
                found = False
                for route in known_routes:
                    if endpoint == route:
                        found = True
                        break
                    # Check for /api/ prefix mismatch which is common
                    if endpoint.startswith("/") and route.endswith(endpoint):
                        found = True
                        break
                    if route.startswith("/api") and endpoint in route:
                        found = True # weak match
                        break
                
                if not found:
                    log(f"[MISMATCH] {os.path.basename(file)} calls '{endpoint}' but no backend route explicitly matches.")

if __name__ == "__main__":
    main()
