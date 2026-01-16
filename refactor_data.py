import sys
import re

def refactor_data_manager(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Ranges to delete (1-indexed, inclusive)
    ranges_to_delete = [
        (3106, 3127), # fetch_prices_loop
        (3128, 3150), # preload_all_ohlcv
        (3151, 3186), # fetch_ohlcv_loop
        (3187, 3245), # fetch_orderbook_loop
    ]
    
    to_remove = set()
    for start, end in ranges_to_delete:
        for i in range(start - 1, end):
            to_remove.add(i)
            
    new_lines = []
    for i, line in enumerate(lines):
        if i not in to_remove:
            # Replace references
            line = line.replace("self.market", "self.data.market")
            line = line.replace("self.ohlcv_buffer", "self.data.ohlcv_buffer")
            line = line.replace("self.orderbook", "self.data.orderbook")
            line = line.replace("self.data_updated_event", "self.data.data_updated_event")
            line = line.replace("self._last_kline_ts", "self.data._last_kline_ts")
            line = line.replace("self._last_kline_ok_ms", "self.data._last_kline_ok_ms")
            line = line.replace("self._preloaded", "self.data._preloaded")
            line = line.replace("self._last_feed_ok_ms", "self.data._last_feed_ok_ms")
            
            # Update main calls
            line = line.replace("orchestrator.fetch_prices_loop()", "orchestrator.data.fetch_prices_loop()")
            line = line.replace("orchestrator.fetch_ohlcv_loop()", "orchestrator.data.fetch_ohlcv_loop()")
            line = line.replace("orchestrator.fetch_orderbook_loop()", "orchestrator.data.fetch_orderbook_loop()")
            line = line.replace("orchestrator.preload_all_ohlcv(", "orchestrator.data.preload_all_ohlcv(")
            
            new_lines.append(line)

    content = "".join(new_lines)
    
    # Add import
    content = content.replace("from core.dashboard_server import DashboardServer", 
                              "from core.dashboard_server import DashboardServer\nfrom core.data_manager import DataManager")

    # Update __init__
    # Remove old data fields (using the already replaced names)
    content = re.sub(r'        self\.data\.market = \{s: \{"price": None, "ts": 0\} for s in SYMBOLS\}\n', '', content)
    content = re.sub(r'        self\.data\.ohlcv_buffer = \{s: deque\(maxlen=OHLCV_PRELOAD_LIMIT\) for s in SYMBOLS\}\n', '', content)
    content = re.sub(r'        self\.data\.orderbook = \{s: \{"ts": 0, "ready": False, "bids": \[\], "asks": \[\]\} for s in SYMBOLS\}\n', '', content)
    content = re.sub(r'        self\.data\._last_kline_ts = \{s: 0 for s in SYMBOLS\}.*?\n', '', content)
    content = re.sub(r'        self\.data\._last_kline_ok_ms = \{s: 0 for s in SYMBOLS\}.*?\n', '', content)
    content = re.sub(r'        self\.data\._preloaded = \{s: False for s in SYMBOLS\}\n', '', content)
    content = re.sub(r'        self\.data\._last_feed_ok_ms = 0\n', '', content)
    
    # Add self.data = DataManager(self)
    content = content.replace("self.dashboard = None # Set in main", 
                              "self.dashboard = None # Set in main\n        self.data = DataManager(self)")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    refactor_data_manager(sys.argv[1])
