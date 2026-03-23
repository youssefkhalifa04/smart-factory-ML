import importlib
import os
import sys

_current_dir = os.path.dirname(__file__)
_removed_from_path = False
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
    _removed_from_path = True

supabase_pkg = importlib.import_module("supabase")
create_client = supabase_pkg.create_client
Client = supabase_pkg.Client

if _removed_from_path:
    sys.path.insert(0, _current_dir)
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

sp: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
