import socket
import time

def test_a2s_info():
    # Construct A2S_INFO packet
    query = b'\xff\xff\xff\xffTSource Engine Query\x00'
    
    print("Testing A2S_INFO query...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        sock.sendto(query, ('127.0.0.1', 27015))
        data, addr = sock.recvfrom(2048)
        print(f"Received {len(data)} bytes from {addr}")
        
        if data.startswith(b'\xff\xff\xff\xffI'):
            print("Valid A2S_INFO response header detected.")
            
            # The structure has the Name string starting after the 6 byte header
            # Find the string after the protocol byte
            rest = data[6:]
            server_name_end = rest.find(b'\x00')
            server_name = rest[:server_name_end].decode('utf-8')
            
            rest = rest[server_name_end+1:]
            map_name_end = rest.find(b'\x00')
            map_name = rest[:map_name_end].decode('utf-8')
            
            print(f"Parsed Server Name: {server_name}")
            print(f"Parsed Map Name (Tunnel URL): {map_name}")
            
            if "http://" in map_name or "https://" in map_name:
                print("SUCCESS: Cloudflare tunnel URL found in response.")
            else:
                print("FAIL: No tunnel URL found.")
        else:
            print("Invalid response header:", data[:10])
            
    except socket.timeout:
        print("FAIL: Socket timeout - no response from colonizer.")
    except Exception as e:
        print(f"FAIL: Error during test - {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    test_a2s_info()
