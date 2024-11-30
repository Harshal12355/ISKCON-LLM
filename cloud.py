# This is the entry point for streamlit cloud, don't mess with it.

# --start-- This fixes a streamlit cloud sqlite issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --end--

# Run the thing
from ui import main
main()