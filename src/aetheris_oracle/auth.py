from fastapi import Depends, HTTPException, Request, status


def api_key_dependency(request: Request) -> None:
    expected = request.app.state.api_key
    if not expected:
        return
    provided = request.headers.get("x-api-key")
    if provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")
