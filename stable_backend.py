#!/usr/bin/env python3
"""
Stable Backend Launcher - Handles crashes and restarts automatically.
"""

import subprocess
import time
import sys
import os
import signal
import psutil

def kill_port_8000():
    """Kill any process using port 8000."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == 8000:
                        print(f"🔧 Killing process {proc.info['pid']} ({proc.info['name']}) using port 8000")
                        proc.kill()
                        time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"⚠️  Could not check port 8000: {e}")

def check_system_resources():
    """Check if system has enough resources."""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"💾 Memory: {memory.percent:.1f}% used ({memory.available // (1024**3):.1f}GB available)")
        print(f"🖥️  CPU: {cpu_percent:.1f}% used")
        
        if memory.percent > 90:
            print("⚠️  WARNING: Low memory! This may cause crashes.")
            return False
        
        if cpu_percent > 95:
            print("⚠️  WARNING: High CPU usage! This may cause instability.")
            return False
            
        return True
        
    except Exception as e:
        print(f"⚠️  Could not check system resources: {e}")
        return True

def run_backend_with_monitoring():
    """Run the backend with crash monitoring and auto-restart."""
    
    print("🚀 Stable Gyroidic Backend Launcher")
    print("=" * 40)
    
    # Check system resources
    if not check_system_resources():
        print("⚠️  System resources are low. Backend may be unstable.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Kill any existing process on port 8000
    kill_port_8000()
    
    restart_count = 0
    max_restarts = 3
    
    while restart_count < max_restarts:
        try:
            print(f"\n🌐 Starting backend (attempt {restart_count + 1}/{max_restarts})...")
            
            # Set environment variables to reduce MKL issues
            env = os.environ.copy()
            env['MKL_NUM_THREADS'] = '1'
            env['OMP_NUM_THREADS'] = '1'
            env['NUMEXPR_NUM_THREADS'] = '1'
            
            # Start the backend process
            process = subprocess.Popen([
                sys.executable, 
                'src/ui/diegetic_backend.py'
            ], 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
            )
            
            print(f"✅ Backend started with PID: {process.pid}")
            
            # Monitor the process
            start_time = time.time()
            last_output_time = time.time()
            
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    print(f"❌ Backend process exited with code: {process.returncode}")
                    break
                
                # Read output
                try:
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())
                        last_output_time = time.time()
                        
                        # Check for successful startup
                        if "STATUS: MANIFOLD COHERENT" in output:
                            print("✅ Backend is ready and stable!")
                            print("🌐 Open http://localhost:8000 in your browser")
                            print("⏹️  Press Ctrl+C to stop")
                            
                except Exception as e:
                    print(f"⚠️  Output reading error: {e}")
                
                # Check for timeout (no output for 30 seconds after startup)
                if time.time() - start_time > 30 and time.time() - last_output_time > 30:
                    print("⏰ Backend appears to be hanging. Restarting...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    break
                
                time.sleep(0.1)
            
            # If we get here, the process exited
            runtime = time.time() - start_time
            
            if runtime < 10:
                print(f"❌ Backend crashed quickly ({runtime:.1f}s). This suggests a serious issue.")
                restart_count += 1
            elif runtime < 60:
                print(f"⚠️  Backend ran for {runtime:.1f}s before crashing. Restarting...")
                restart_count += 1
            else:
                print(f"✅ Backend ran successfully for {runtime:.1f}s")
                # Reset restart count for long-running sessions
                restart_count = 0
            
            if restart_count < max_restarts:
                print(f"🔄 Restarting in 3 seconds...")
                time.sleep(3)
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping backend...")
            if 'process' in locals():
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
            return 0
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            restart_count += 1
            time.sleep(3)
    
    print(f"❌ Backend failed to start after {max_restarts} attempts.")
    print("🔧 Possible solutions:")
    print("   • Restart your computer")
    print("   • Check if another program is using port 8000")
    print("   • Run: netstat -ano | findstr :8000")
    
    return 1

if __name__ == "__main__":
    try:
        sys.exit(run_backend_with_monitoring())
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)