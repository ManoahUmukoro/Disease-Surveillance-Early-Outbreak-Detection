"""
mongo_store.py — the UNSTRUCTURED data layer (MongoDB) of the Adaptive SVM surveillance system.

In the Chapter 4 architecture this is the "unstructured / semi-structured store": clinical notes,
laboratory information, and uploaded documents (PDF / DOCX / images), each linked to a structured
case in SQLite by case_id.

Graceful degradation: if pymongo isn't installed or no MONGODB_URI is configured, `available` is
False and writes become no-ops — the app still runs entirely on the SQLite path. On Streamlit
Community Cloud, set MONGODB_URI in the app's Secrets (reuse the existing Atlas cluster). Under
Docker Compose a local mongo service is used.
"""
import os
from datetime import datetime

DB_NAME = "surveillance_unstructured"


def _uri():
    uri = os.environ.get("MONGODB_URI") or os.environ.get("ATLAS_MONGODB_URI")
    if not uri:
        try:
            import streamlit as st
            uri = st.secrets.get("MONGODB_URI", None)
        except Exception:
            uri = None
    return uri


class MongoStore:
    """Thin wrapper that never raises on connection failure — check `.available`."""

    def __init__(self):
        self.available = False
        self.reason = ""
        self._db = None
        self._fs = None
        uri = _uri()
        if not uri:
            self.reason = "no MONGODB_URI configured (using SQLite only)"
            return
        try:
            from pymongo import MongoClient
            import gridfs
            client = MongoClient(uri, serverSelectionTimeoutMS=4000)
            client.admin.command("ping")           # fail fast if unreachable
            self._db = client[DB_NAME]
            self._fs = gridfs.GridFS(self._db)
            self.available = True
        except Exception as e:
            self.reason = f"{type(e).__name__}: {e}"

    def save_case_documents(self, case_id, clinical_notes="", lab_info=None, files=None):
        """Persist the unstructured part of a case.

        files: list of (filename, bytes, content_type). Large files go to GridFS; a case_records
        document holds the notes, lab info, and file metadata. Returns a summary dict.
        """
        if not self.available:
            return {"stored": False, "reason": self.reason}
        doc = {
            "case_id": case_id,
            "clinical_notes": clinical_notes or "",
            "lab_info": lab_info or {},
            "created_at": datetime.utcnow().isoformat(),
            "files": [],
        }
        for name, data, ctype in (files or []):
            fid = self._fs.put(data, filename=name, contentType=ctype, case_id=case_id)
            doc["files"].append({"filename": name, "content_type": ctype,
                                 "size": len(data), "gridfs_id": str(fid)})
        self._db["case_records"].insert_one(doc)
        return {"stored": True, "db": DB_NAME, "collection": "case_records",
                "n_files": len(doc["files"])}

    def get_case_record(self, case_id):
        if not self.available:
            return None
        return self._db["case_records"].find_one({"case_id": case_id}, {"_id": 0})

    def stats(self):
        if not self.available:
            return {"available": False, "reason": self.reason}
        return {"available": True, "db": DB_NAME,
                "records": self._db["case_records"].estimated_document_count(),
                "files": self._db["fs.files"].estimated_document_count()}
