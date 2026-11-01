"""Bounded diagnostics for rejected requests; never replay partial generations."""
from __future__ import annotations

import json
import math
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime


def retry_after(value):
    if isinstance(value,bool) or value is None:return None
    try:seconds=float(value)
    except (ValueError,TypeError):
        try:
            date=parsedate_to_datetime(str(value))
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
            seconds=(date-datetime.now(timezone.utc)).total_seconds()
        except (ValueError,TypeError,OverflowError):return None
    return max(0,seconds) if math.isfinite(seconds) and seconds>=0 else None


def rejection_details(response,key):
    try:body=response.json()
    except (ValueError,TypeError):body={}
    error=body.get('error',{}) if isinstance(body,dict) else {}
    if not isinstance(error,dict):error={}
    metadata=error.get('metadata')
    if not isinstance(metadata,dict):metadata={}
    message=str(error.get('message') or 'Request rejected')
    raw=metadata.get('raw')
    if isinstance(raw,str):
        try:
            detail=json.loads(raw)
            nested=detail.get('error',{}) if isinstance(detail,dict) else {}
            raw=str(nested.get('message','')) if isinstance(nested,dict) else ''
        except (ValueError,TypeError):pass
        if raw:message+=': '+raw
    def redact(value,limit):
        text=str(value)
        return (text.replace(key,'[redacted]') if key else text)[:limit]
    result={'status':response.status_code,'message':redact(message,600)}
    # Do not persist full provider metadata, request bodies, headers or user IDs.
    for name in ('provider_name','provider_error_code','error_type','limit_source'):
        if isinstance(metadata.get(name),str):result[name]=redact(metadata[name],120)
    delay=retry_after(response.headers.get('retry-after'))
    if delay is None:delay=retry_after(metadata.get('retry_after_seconds'))
    if delay is not None:result['retry_after_seconds']=delay
    return result
