import streamlit as st
import pyrebase
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import uuid
import math

firebaseConfig = {
    'apiKey': "AIzaSyAzXYx4LsVfV7I5E7tL35rVUdzlFGQqLKU",
    'authDomain': "mte-website-best.firebaseapp.com",
    'databaseURL': "https://mte-website-best-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "mte-website-best",
    'storageBucket': "mte-website-best.appspot.com",
    'messagingSenderId': "680810824940",
    'appId': "1:680810824940:web:70f8311fda11c3981211c9",
    'measurementId': "G-XYEN19L1EL"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_data" not in st.session_state:
    st.session_state.user_data = None
if "time_range" not in st.session_state:
    st.session_state.time_range = "All Time"

def cleanemail(email: str) -> str:
    return email.replace("@", "_").replace(".", "_") if email else ""

def _normalize_record(rec: dict) -> dict:
    if not isinstance(rec, dict):
        return {}
    out = dict(rec)
    if "Timestamp" not in out:
        for k in list(out.keys()):
            if k.lower() in ("timestamp", "time", "ts", "date", "datetime"):
                out["Timestamp"] = out.pop(k)
                break
    if "Timestamp" in out:
        val = out["Timestamp"]
        try:
            if isinstance(val, (int, float)):
                if val > 1e10:
                    out["Timestamp"] = datetime.utcfromtimestamp(val/1000).isoformat()
                else:
                    out["Timestamp"] = datetime.utcfromtimestamp(val).isoformat()
            else:
                out["Timestamp"] = str(val)
        except:
            out["Timestamp"] = str(val)
    for axis in ("x", "y", "angle"):
        if axis in out:
            try:
                out[axis] = float(out[axis])
            except:
                out.pop(axis, None)
    return out

def get_last_1000(email: str) -> pd.DataFrame:
    if not email:
        return pd.DataFrame()
    cleaned_email = cleanemail(email)
    try:
        query = db.child("users").child(cleaned_email).child("sensor_data").order_by_key().limit_to_last(1000).get()
    except:
        return pd.DataFrame()
    items = []
    if query.each():
        for d in query.each():
            rec = _normalize_record(d.val())
            if "Timestamp" not in rec:
                rec["Timestamp"] = datetime.utcnow().isoformat()
            items.append(rec)
    return pd.DataFrame(items) if items else pd.DataFrame()

def filter_data(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or "Timestamp" not in df.columns:
        return df
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    now = datetime.now()
    if period == "Today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "1 Month":
        start = now - timedelta(days=30)
    elif period == "6 Months":
        start = now - timedelta(days=180)
    elif period == "8 Months":
        start = now - timedelta(days=240)
    else:
        return df
    return df[df["Timestamp"] >= start]

def monotonic_chain(points):
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower) >=2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper) >=2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def centroid(points):
    if not points:
        return (0,0)
    A = 0; Cx = 0; Cy = 0
    n = len(points)
    for i in range(n):
        x0,y0 = points[i]
        x1,y1 = points[(i+1)%n]
        cross = x0*y1 - x1*y0
        A += cross
        Cx += (x0+x1)*cross
        Cy += (y0+y1)*cross
    A = A/2
    if abs(A) < 1e-9:
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    return (Cx/(6*A), Cy/(6*A))

def sample_radial_signature(polygon, n_angles=180):
    if not polygon or len(polygon) < 2:
        return np.zeros(n_angles)
    cx,cy = centroid(polygon)
    edges=[]
    for i in range(len(polygon)):
        edges.append((polygon[i], polygon[(i+1)%len(polygon)]))
    radii = np.zeros(n_angles)
    angles = np.linspace(0,2*math.pi,n_angles,endpoint=False)
    for idx,ang in enumerate(angles):
        dx = math.cos(ang); dy = math.sin(ang)
        t_hits=[]
        for (x1,y1),(x2,y2) in edges:
            ex = x2-x1; ey = y2-y1
            denom = dx*ey - dy*ex
            if abs(denom) < 1e-9: continue
            t = ((x1-cx)*ey - (y1-cy)*ex)/denom
            u = ((x1-cx)*dy - (y1-cy)*dx)/denom
            if t>=0 and 0<=u<=1:
                t_hits.append(t)
        radii[idx] = min(t_hits) if t_hits else 0.0
    return radii

def signature_similarity(sig1, sig2):
    max_val = max(sig1.max(), sig2.max(), 1e-6)
    diff = np.mean(np.abs(sig1 - sig2)) / max_val
    return max(0.0, 1.0 - diff)

def load_classes(email):
    cleaned = cleanemail(email)
    try:
        classes = db.child("users").child(cleaned).child("classes").get()
    except:
        return {}
    result = {}
    if classes.each():
        for c in classes.each():
            result[c.key()] = c.val()
    return result

def save_new_class(email, class_name, polygon_points):
    cleaned = cleanemail(email)
    class_obj = {"name": class_name, "created_at": datetime.utcnow().isoformat(), "example_points": polygon_points}
    return db.child("users").child(cleaned).child("classes").push(class_obj)

def store_instance_in_class(email, class_key, polygon_points, score):
    cleaned = cleanemail(email)
    unique_id = str(uuid.uuid4())[:8]
    instance = {"id": unique_id, "timestamp": datetime.utcnow().isoformat(), "points": polygon_points, "score": float(score)}
    db.child("users").child(cleaned).child("classes").child(class_key).child("instances").push(instance)
    return unique_id

def signup():
    st.header("Sign Up")
    email = st.text_input("Enter your email", key="signup_email").strip()
    password = st.text_input("Enter your password", type="password", key="signup_pass").strip()
    if st.button("Sign Up"):
        try:
            auth.create_user_with_email_and_password(email, password)
            db.child("users").child(cleanemail(email)).child("profile").set({"username": email})
            st.success("Account created successfully. Please log in.")
        except Exception as e:
            st.error(f"Error in signup: {e}")

def login():
    st.header("Login")
    email = st.text_input("Enter your email", key="login_email").strip()
    password = st.text_input("Enter your password", type="password", key="login_pass").strip()
    if st.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_data = user
            st.success("Login Successful.")
        except Exception as e:
            st.error(f"Error in login: {e}")

def user_page():
    st.set_page_config(page_title="MTE Dashboard", layout="wide")
    user_email = st.session_state.user_email
    username = cleanemail(user_email)
    st.sidebar.markdown("<h2 style='color:#5BC0DE;'>MTE Dashboard</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>User:</b> {username}", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", ["Home", "Live Contour", "Historical Plots", "Classes", "Logout"])
    st.markdown("<style> body { background-color: #0B0C10; color: #C5C6C7; } .stButton>button { background-color: #45A29E; color: white; border-radius: 8px; font-weight: bold; } </style>", unsafe_allow_html=True)

    if page == "Home":
        st.title(f"Welcome, {user_email}")
        st.write("ESP32 Data Status Monitor")
    
    elif page == "Logout":
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.user_data = None
            st.success("Logged out successfully.")

    elif page == "Live Contour":
        st.title("Live Ultrasonic Points → Contour")
        st_autorefresh(interval=3000, key="live_contour_refresh")
        df = get_last_1000(user_email)
        if df.empty:
            st.info("No sensor data available.")
            return

        if "x" not in df.columns or "y" not in df.columns:
            st.error("Missing x,y columns.")
            st.write(df.head(10))
            return

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x","y"])
        if df.empty:
            st.info("No valid points.")
            return

        recent = df.tail(500)
        points = list(zip(recent['x'], recent['y']))
        hull = monotonic_chain(points)
        hull_x = [p[0] for p in hull] + ([hull[0][0]] if hull else [])
        hull_y = [p[1] for p in hull] + ([hull[0][1]] if hull else [])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent['x'], y=recent['y'], mode='markers', name='Points'))
        if hull:
            fig.add_trace(go.Scatter(x=hull_x, y=hull_y, fill='toself', name='Contour', line=dict(width=2)))
        fig.update_layout(title="Live Contour", template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        if len(hull) >= 3:
            sig_live = sample_radial_signature(hull, n_angles=180)
            classes = load_classes(user_email)
            best = {"key": None, "score": 0, "name": None}
            for key, c in classes.items():
                example = c.get("example_points")
                if not example:
                    continue
                ex_hull = monotonic_chain([tuple(pt) for pt in example])
                if len(ex_hull) < 3: 
                    continue
                sig_ex = sample_radial_signature(ex_hull, 180)
                sim = signature_similarity(sig_live, sig_ex)
                if sim > best["score"]:
                    best = {"key": key, "score": sim, "name": c.get("name")}

            st.subheader("Matching result")
            threshold = st.slider("Match threshold (%)", 10, 95, 60, 5)
            if best["key"] and best["score"]*100 >= threshold:
                percent = best["score"]*100
                st.success(f"Matched: {best['name']} ({percent:.1f}%)")
                if st.button("Store this instance"):
                    uid = store_instance_in_class(user_email, best["key"], hull, float(percent))
                    st.info(f"Stored instance: {uid}")
            else:
                new_name = st.text_input("New class name")
                if st.button("Create new class"):
                    if new_name.strip() == "":
                        st.error("Enter a name.")
                    else:
                        save_new_class(user_email, new_name.strip(), hull)
                        st.success(f"Class '{new_name}' created.")

        st.subheader("Raw Latest Points")
        cols = [c for c in ["Timestamp","x","y","angle"] if c in recent.columns]
        st.dataframe(recent.tail(50)[cols].reset_index(drop=True))

    elif page == "Historical Plots":
        st.title("Historical Data")
        df = get_last_1000(user_email)
        if df.empty:
            st.info("No data.")
            return

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("Today"): st.session_state.time_range = "Today"
        with col2:
            if st.button("1 Month"): st.session_state.time_range = "1 Month"
        with col3:
            if st.button("6 Months"): st.session_state.time_range = "6 Months"
        with col4:
            if st.button("8 Months"): st.session_state.time_range = "8 Months"
        with col5:
            if st.button("All Time"): st.session_state.time_range = "All Time"

        filtered_df = filter_data(df, st.session_state.time_range)
        if not filtered_df.empty and "x" in filtered_df.columns and "y" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("Timestamp")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df["x"], y=filtered_df["y"], mode='markers+lines'))
            fig.update_layout(title=f"Trajectory ({st.session_state.time_range})", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Filtered data missing x/y.")

    elif page == "Classes":
        st.title("Stored Classes")
        classes = load_classes(user_email)
        if not classes:
            st.info("No classes found.")
            return
        for key, c in classes.items():
            st.subheader(f"{c.get('name')} (key: {key})")
            example = c.get("example_points") or []
            if example:
                ex_hull = monotonic_chain([tuple(pt) for pt in example])
                ex_x = [p[0] for p in ex_hull] + ([ex_hull[0][0]] if ex_hull else [])
                ex_y = [p[1] for p in ex_hull] + ([ex_hull[0][1]] if ex_hull else [])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[pt[0] for pt in example], y=[pt[1] for pt in example], mode='markers'))
                if ex_hull:
                    fig.add_trace(go.Scatter(x=ex_x, y=ex_y, fill='toself'))
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
            instances = c.get("instances") or {}
            if instances:
                rows=[]
                if isinstance(instances, dict):
                    for inst in instances.values():
                        rows.append({"id": inst.get("id"), "timestamp": inst.get("timestamp"), "score": inst.get("score")})
                st.table(pd.DataFrame(rows))
            else:
                st.write("No stored instances.")

st.set_page_config(page_title="MTE Dashboard", layout="wide")
st.title("MTE Project Dashboard — Contour Mode")

if st.session_state.logged_in:
    user_page()
else:
    auth_choice = st.selectbox("Select Action", ["Login", "Sign Up"])
    if auth_choice == "Sign Up":
        signup()
    else:
        login()
