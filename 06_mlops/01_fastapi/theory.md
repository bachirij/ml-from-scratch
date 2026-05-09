# FastAPI & REST APIs — Theory

---

## 1. Client / Server

When you interact with a web application, there are always two actors:

- **The client**: the one who **requests**. In a browser, that's you typing a URL. In your project, it will be curl, FastAPI's `/docs` page, or later a Streamlit frontend.
- **The server**: the one who **receives the request, processes it, and responds**. In your project, that's the FastAPI application running on your machine.

The client and server communicate via a shared protocol: **HTTP**.

---

## 2. HTTP — The Communication Protocol

**HTTP** (HyperText Transfer Protocol) is the protocol that defines how messages flow between client and server.

Every exchange follows the same pattern:

```
Client  ──── request ────►  Server
Client  ◄─── response ───   Server
```

- The **request** originates from the client: it contains the type of operation, the target address, and optionally some data.
- The **response** originates from the server: it contains a status code (200 = OK, 404 = not found, etc.) and optionally some data.

---

## 3. GET vs POST

HTTP defines several **methods** (also called verbs) that indicate the intent of the request.

| Method   | Intent                     | Example in your project                          |
| -------- | -------------------------- | ------------------------------------------------ |
| **GET**  | Read / retrieve a resource | `/health` — check that the API is running        |
| **POST** | Send data for processing   | `/predict` — send features, receive a prediction |

**Important rule**: GET does not modify anything on the server. POST sends something that the server will use to perform an operation.

---

## 4. Endpoint

An **endpoint** is a specific address on the server, associated with a particular piece of functionality.

Concretely in your project:

```
http://localhost:8000/health   ← endpoint 1: API status
http://localhost:8000/predict  ← endpoint 2: model prediction
```

Each endpoint responds to a specific HTTP method (GET or POST) and does exactly one well-defined thing.

---

## 5. JSON

**JSON** (JavaScript Object Notation) is the standard format for exchanging data between client and server.

It is structurally very close to a Python dictionary:

```json
{
  "feature_1": 3.5,
  "feature_2": 12.0
}
```

Python equivalent:

```python
{"feature_1": 3.5, "feature_2": 12.0}
```

**Why JSON?** It is plain text, so it is universal. Whether the client is written in Python, JavaScript, or anything else — everyone can read and write JSON.

In your project:

- The client sends the **features** as JSON in the body of the POST request
- The server returns the **prediction** as JSON in the response

---

## 6. REST

**REST** (Representational State Transfer) is a set of conventions for designing HTTP APIs in a consistent and readable way.

It is not a technology — it is an architectural style. The two most important rules:

**Rule 1 — URLs identify resources, not actions**

```
/predict      ← a resource
/doPredict    ← an action (non-REST style)
```

**Rule 2 — The HTTP verb carries the intent**

```
GET    → read
POST   → send for processing / create
PUT    → modify
DELETE → delete
```

An API that follows these conventions is called **RESTful**. Your FastAPI with `/health` (GET) and `/predict` (POST) is a minimal REST API.

---

## 7. Full Flow for `/predict`

To ground everything above in a concrete example:

```
1. Client sends a POST request to http://localhost:8000/predict
   Request body (JSON): {"feature_1": 3.5, "feature_2": 12.0}

2. Server (FastAPI) receives the request
   → Validates the data (Pydantic)
   → Loads the LinearRegression model
   → Calls model.predict(X)

3. Server returns a response (JSON): {"prediction": 42.7}

4. Client receives the response and can display / use it
```

---

## 8. Uvicorn — The Server That Runs FastAPI

FastAPI is a Python framework — it defines your endpoints, validates data, and builds responses. But this Python code alone cannot "run" on a network. It needs something that **listens** for incoming network connections and forwards them to your application.

That is uvicorn's role. It acts as the bridge between the network (incoming HTTP requests) and your FastAPI application.

An analogy: FastAPI is the cook, uvicorn is the restaurant — it opens the door, receives customers, and delivers the dishes the cook prepared.

**How to start your API:**

```bash
uvicorn main:app --reload
```

Breaking this down:

- `main` → the file `main.py`
- `app` → the `app` variable inside that file (your FastAPI object)
- `--reload` → automatically restarts the server when you save changes (useful during development)

**Why not just `python main.py`?**

When Python executes a normal script, it reads it top to bottom and stops. It does not know how to:

- **Stay active** and wait for incoming connections
- **Handle multiple requests** arriving at the same time
- **Speak ASGI** — the standard protocol that allows FastAPI to communicate with the outside world

Uvicorn handles all of this. You do not need to understand ASGI in depth right now. The key takeaway: **FastAPI defines the logic, uvicorn runs it on the network. Both are required.**

---

## 9. Project Tools

| Tool               | Role                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| **FastAPI**        | The web framework — defines endpoints and handles requests/responses |
| **uvicorn**        | The ASGI server that runs the FastAPI application                    |
| **Pydantic**       | Input/output data validation (built into FastAPI)                    |
| **joblib**         | Loading the saved `.pkl` model                                       |
| **curl / `/docs`** | Manually testing the endpoints                                       |

---

## References

- [FastAPI — Official Documentation](https://fastapi.tiangolo.com/)
- [HTTP Methods — MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)
- [JSON — MDN](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON)
