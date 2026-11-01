import os
os.environ["ENVIRONMENT"] = "test"

import sys

# Clear any cached modules that might have imported integration models before conftest runs
modules_to_clear = [k for k in sys.modules.keys() if k.startswith('core.database.models.integration')]
for mod in modules_to_clear:
    del sys.modules[mod]

# Also clear main app modules that might have imported integration models
modules_to_clear = [k for k in sys.modules.keys() if k.startswith('main')]
for mod in modules_to_clear:
    del sys.modules[mod]

# Clear config and database config modules to force re-import with test environment
for mod_name in ['core.config', 'core.database.config', 'core.database.test_config']:
    if mod_name in sys.modules:
        del sys.modules[mod_name]