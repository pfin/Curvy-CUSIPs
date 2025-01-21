import shelve
import os 

class ShelveDBWrapper:
    def __init__(self, db_path, create=False):
        """Initialize shelve database wrapper.
        
        Args:
            db_path: Path to the shelve database (without extensions)
            create: If True, create new database if it doesn't exist
        """
        self.db_path = db_path
        self.db = None
        
        # Only check if database exists when not creating
        if not create:
            try:
                # Try opening briefly to check existence
                with shelve.open(db_path, flag='r') as _:
                    pass
            except:
                raise ValueError(f"Database does not exist at {db_path}")

    def open(self):
        """Open the shelve database."""
        self.db = shelve.open(self.db_path)
        return self.db

    def close(self):
        """Close the shelve database."""
        if self.db is not None:
            self.db.close()

    def get(self, key):
        """Retrieve a value by key."""
        if self.db is not None:
            return self.db[key]
        raise RuntimeError("Database is not open.")

    def set(self, key, value):
        """Set a value by key."""
        if self.db is not None:
            self.db[key] = value
        else:
            raise RuntimeError("Database is not open.")

    def keys(self):
        if self.db is not None:
            return self.db.keys() 
        else:
            raise RuntimeError("Database is not open.")
        
    def exists(self, key):
        """Check if a key exists in the database."""
        if self.db is not None:
            return key in self.db
        else:
            raise RuntimeError("Database is not open.")
        
    def delete(self, key):
        """Delete a key from the database."""
        if self.db is not None:
            if key in self.db:
                del self.db[key]
            else:
                raise KeyError(f"Key '{key}' does not exist in the database.")
        else:
            raise RuntimeError("Database is not open.")