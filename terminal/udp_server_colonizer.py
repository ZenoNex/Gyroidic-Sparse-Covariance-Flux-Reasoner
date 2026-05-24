import socket
import threading
import time
import logging
import struct

logger = logging.getLogger("UDPServerColonizer")

class OptionD_Colonizer:
    def __init__(self, port=27015, app_id=320): # 320 is HL2: Deathmatch
        self.port = port
        self.app_id = app_id
        self.tunnel_url = "Waiting for Tunnel..."
        self.running = False
        self.sock = None
        self.server_name = "GYROIDIC_FLUX_ANOMALY"

    def set_tunnel_url(self, url):
        self.tunnel_url = url
        logger.info(f"UDP Colonizer updated with tunnel URL: {url}")

    def _pack_string(self, s):
        """Packs a null-terminated string for Source protocol"""
        return s.encode('utf-8') + b'\x00'

    def _build_a2s_info_response(self):
        """Constructs a valid A2S_INFO response packet"""
        payload = bytearray()
        payload.extend(b'\xff\xff\xff\xff') # Split packet header
        payload.append(0x49) # 'I' - A2S_INFO response header
        payload.append(0x11) # Protocol version (17)
        
        # Packing fields
        payload.extend(self._pack_string(f"[{self.server_name}] {self.tunnel_url}")) # Server Name
        payload.extend(self._pack_string(self.tunnel_url)) # Map Name
        payload.extend(self._pack_string("gyroidic_resonance")) # Folder
        payload.extend(self._pack_string("Gyroidic Reasoner Option D")) # Game
        
        payload.extend(struct.pack('<H', self.app_id)) # App ID (short)
        
        payload.append(1) # Players
        payload.append(1) # Max Players
        payload.append(0) # Bots
        
        payload.append(ord('d')) # Server Type (dedicated)
        payload.append(ord('w')) # Environment (windows)
        payload.append(0) # Visibility (public)
        payload.append(0) # VAC (unsecured)
        
        payload.extend(self._pack_string("1.0.0.0")) # Version
        payload.append(0) # Extra Data Flag
        
        return bytes(payload)

    def _listen_loop(self):
        """Listens for incoming A2S_INFO queries and responds"""
        logger.info(f"UDP Colonizer listening on port {self.port}...")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data.startswith(b'\xff\xff\xff\xffTSource Engine Query'):
                    # Respond with A2S_INFO
                    resp = self._build_a2s_info_response()
                    self.sock.sendto(resp, addr)
                    logger.debug(f"Answered A2S_INFO query from {addr}")
            except socket.timeout:
                continue
            except (socket.error, OSError) as e:
                # If we closed the socket intentionally during shutdown, suppress the error
                if not self.running:
                    break
                logger.error(f"Error in UDP Colonizer listener: {e}")
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Error in UDP Colonizer listener: {e}")

    def _heartbeat_loop(self):
        """Sends periodic heartbeats to the master server list"""
        master_servers = [
            ("hl2master.steampowered.com", 27011),
            ("hl2master.steampowered.com", 27015),
        ]
        
        logger.info("UDP Colonizer heartbeat thread started.")
        while self.running:
            for master in master_servers:
                try:
                    # q (0x71) followed by connection info or just bare packet
                    heartbeat_packet = b'\xff\xff\xff\xffq\x00'
                    self.sock.sendto(heartbeat_packet, master)
                    logger.debug(f"Sent heartbeat to {master}")
                except Exception as e:
                    logger.error(f"Failed to heartbeat to {master}: {e}")
            
            # Wait 60 seconds before next heartbeat
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(1)

    def start(self):
        if self.running: return
        self.running = True
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', self.port))
            self.sock.settimeout(1.0)
            
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            
            self.listener_thread.start()
            self.heartbeat_thread.start()
            logger.info("Option D UDP Colonizer successfully started.")
        except Exception as e:
            logger.error(f"Failed to start UDP Colonizer: {e}")
            self.running = False

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        logger.info("Option D UDP Colonizer stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    colonizer = OptionD_Colonizer()
    colonizer.set_tunnel_url("")
    colonizer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        colonizer.stop()
