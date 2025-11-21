import socket, time, random, os, json, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv

def log_epoch(epoch, actions, rewards, reactions, filename="run_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Action", "Reward", "Reactions"])
        for action, reward in zip(actions, rewards):
            writer.writerow([epoch, action, reward, reactions])

TARGET = "127.0.0.1"

# ------------------- Utility Modules -------------------
def fast_scan(host, ports=range(1, 1000), timeout=0.3, workers=200):
    open_ports = []
    def scan_port(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((host, port)) == 0:
                    return port
        except: return None
        return None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_port, p) for p in ports]
        for fut in as_completed(futures):
            port = fut.result()
            if port: open_ports.append(port)
    return open_ports

def stealth_scan(host, ports=range(1, 1000), timeout=1.0, workers=20, delay=0.5):
    open_ports = []
    def scan_port(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((host, port)) == 0:
                    return port
        except: return None
        return None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_port, p) for p in ports]
        for fut in as_completed(futures):
            port = fut.result()
            if port: open_ports.append(port)
            time.sleep(delay)
    return open_ports

def banner_grabber(host, ports):
    banners = {}
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((host, port)) == 0:
                    s.sendall(b"HELLO\r\n")
                    data = s.recv(128).decode(errors="ignore").strip()
                    banners[port] = data
        except: continue
    return banners

FINGERPRINTS = {"SSH": r"^SSH","HTTP": r"HTTP","FTP": r"FTP"}
def fingerprint_services(banners):
    services = set()
    for banner in banners.values():
        for service, pattern in FINGERPRINTS.items():
            if re.search(pattern, banner, re.I):
                services.add(service)
    return services

# ------------------- Post-Ex Emulator -------------------
def simulate_lateral_movement(proto="SSH"):
    print(f"[*] Simulated lateral probe via {proto}")
    return True
def simulate_file_enumeration():
    files = ["confidential.txt","finance_report.doc"]
    print(f"[*] Simulated file enumeration: {files}")
    return files
def simulate_credential_theft():
    creds = ["user:pass123","admin:admin"]
    print(f"[*] Simulated credential theft: {creds}")
    return creds
def simulate_exfil():
    print("[*] Simulated exfiltration of test token")
    return True

# ------------------- Reaction Tracker (Week 13) -------------------
class ReactionTracker:
    def __init__(self):
        self.timeouts = 0
        self.refused = 0
        self.decoys = 0
    def log_result(self, status):
        if status == "timeout": self.timeouts += 1
        elif status == "refused": self.refused += 1
        elif status == "decoy": self.decoys += 1
    def detect_behavior(self):
        if self.refused > 20: return "firewall"
        elif self.timeouts > 10: return "rate-limit"
        elif self.decoys > 0: return "ids"
        return None

# ------------------- RL Policy -------------------
class BanditPolicy:
    def __init__(self, actions, epsilon=0.3, decay=0.99):
        self.actions = actions
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.epsilon = epsilon
        self.decay = decay
    def choose(self, allowed_actions):
        if random.random() < self.epsilon:
            return random.choice(allowed_actions)
        return max(allowed_actions, key=lambda a: self.Q[a])
    def update(self, action, reward):
        self.N[action] += 1
        n = self.N[action]
        self.Q[action] += (reward - self.Q[action]) / n
        self.epsilon = max(0.05, self.epsilon * self.decay)

def save_policy(policy, filename="policy.json"):
    data = {"Q": dict(policy.Q),"N": dict(policy.N),"epsilon": policy.epsilon}
    with open(filename,"w") as f: json.dump(data,f)
def load_policy(policy, filename="policy.json"):
    if os.path.exists(filename):
        with open(filename) as f: data = json.load(f)
        policy.Q.update(data["Q"]); policy.N.update(data["N"]); policy.epsilon = data["epsilon"]

# ------------------- ML Predictor (Week 12) -------------------
class AttackPredictor:
    def __init__(self, logfile="run_log.csv"):
        self.clf = DecisionTreeClassifier()

        # Try to load real run data
        if os.path.exists(logfile):
            X, y = [], []
            with open(logfile) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Features: action type, reward, reactions
                    action = row["Action"]
                    reward = float(row["Reward"])
                    reactions = int(row["Reactions"])

                    # Encode action as simple numeric category
                    action_idx = ACTIONS.index(action) if action in ACTIONS else -1
                    if action_idx >= 0:
                        X.append([action_idx, reward, reactions])
                        # Label: success if reward > 0.2
                        y.append(1 if reward > 0.2 else 0)

            if X:
                self.clf.fit(X, y)
                print(f"[*] Predictor retrained on {len(X)} samples")
            else:
                self._train_dummy()
        else:
            self._train_dummy()

    def _train_dummy(self):
        # Fallback dummy training if no log exists
        X = [[1,1,0],[1,0,1],[0,0,0],[1,1,1]]
        y = [1,0,0,0]
        self.clf.fit(X,y)
        print("[*] Predictor trained on dummy dataset")

    def predict(self, open_ports, services, reactions):
        # Simplified feature: just use counts
        features = np.array([[len(open_ports), len(services), reactions]])
        return self.clf.predict_proba(features)[0][1]

# ------------------- Attack Graph (Week 14) -------------------
class AttackGraph:
    def __init__(self):
        self.edges = defaultdict(lambda: {"w": 0.0, "n": 0})
        self.nodes = set()
        self.prereqs = defaultdict(set)
    def add_node(self, node): self.nodes.add(node)
    def add_edge(self, u, v, init_w=0.0, prereq_flags=None):
        self.add_node(u); self.add_node(v)
        self.edges[(u,v)] = {"w": init_w, "n": 0}
        if prereq_flags: self.prereqs[v].update(prereq_flags)
    def allowed(self, v, state_flags):
        req = self.prereqs.get(v, set())
        return req.issubset(state_flags)
    def update_edge(self, u, v, reward, lr=0.2):
        e = self.edges[(u,v)]
        e["n"] += 1
        e["w"] = e["w"] + lr*(reward - e["w"])
    def best_path_layered(self, layers, state_flags):
        path, total = [], 0.0
        for layer in layers:
            candidates=[]
            for node in layer:
                if self.allowed(node, state_flags):
                    if path:
                        u=path[-1]
                        w=self.edges.get((u,node),{"w":-1.0})["w"]
                    else:
                        w=self.edges.get(("START",node),{"w":0.0})["w"]
                    candidates.append((node,w))
            if candidates:
                node,w=max(candidates,key=lambda x:x[1])
                path.append(node); total+=w
        return path,total
    def save(self, filename="attack_graph.json"):
        data={"edges":{f"{u}|{v}":e for (u,v),e in self.edges.items()},
              "nodes":list(self.nodes),
              "prereqs":{k:list(v) for k,v in self.prereqs.items()}}
        with open(filename,"w") as f: json.dump(data,f)
    def load(self, filename="attack_graph.json"):
        try:
            with open(filename) as f: data=json.load(f)
            self.nodes=set(data["nodes"])
            self.prereqs={k:set(v) for k,v in data["prereqs"].items()}
            self.edges.clear()
            for key,e in data["edges"].items():
                u,v=key.split("|",1)
                self.edges[(u,v)]=e
        except FileNotFoundError: pass

# ------------------- Decision Engine -------------------
ACTIONS = ["FastScan","StealthScan","BannerGrab","HTTPProbe",
           "LateralProbeSSH","LateralProbeSMB",
           "EnumerateFiles","CredTheft","ExfilTestToken"]

class DecisionEngine:
    def __init__(self, target=TARGET, personality="opportunistic"):
        self.policy = BanditPolicy(ACTIONS, epsilon=0.3, decay=0.995)
        load_policy(self.policy)
        self.predictor = AttackPredictor()
        self.tracker = ReactionTracker()
        self.graph = AttackGraph(); self.graph.load()
        self.personality = personality
        self.target = target
        self.state = {"open_ports": [], "services": set(), "reactions": 0}
        self.state["last_action"] = []

    def allowed_actions(self, phase):
        if phase == "Recon":
            return ["FastScan", "StealthScan"]

        if phase == "Fingerprinting":
            return ["BannerGrab", "HTTPProbe"] if self.state["open_ports"] else []

        if phase == "PostEx":
            allowed = []
            if "SSH" in self.state["services"]:
                allowed.append("LateralProbeSSH")
            if "HTTP" in self.state["services"]:
                allowed.append("LateralProbeSMB")
            allowed += ["EnumerateFiles", "CredTheft"]
            return allowed

        if phase == "Exfil":
            return ["ExfilTestToken"]

        return []

    def run_phase(self, phase):
        actions = self.allowed_actions(phase)
        if not actions:
            return

        # Combine Q-values and ML predictions
        scores = {}
        for a in actions:
            q = self.policy.Q[a]
            p = self.predictor.predict(self.state["open_ports"], self.state["services"], self.state["reactions"])
            scores[a] = 0.6*q + 0.4*p

        # Choose best action
        action = max(scores, key=scores.get)
        reward = self.execute(action)
        self.policy.update(action, reward)

        # Update AttackGraph edges
        if self.state["last_action"]:
            prev = self.state["last_action"][0]
            self.graph.update_edge(prev, action, reward)
        self.state["last_action"] = [action]

        # Check defender behavior
        behavior = self.tracker.detect_behavior()
        if behavior == "firewall":
            print("⚠️ Firewall detected → switching to stealth mode")
        elif behavior == "rate-limit":
            print("⚠️ Rate-limit detected → backing off")
            time.sleep(5)
        elif behavior == "ids":
            print("⚠️ IDS alert → avoiding trap service")

    def execute(self, action):
        if action == "FastScan":
            self.state["open_ports"] = fast_scan(self.target)
            if not self.state["open_ports"]:
                self.tracker.log_result("refused")
            return 1.0 if self.state["open_ports"] else 0.0

        elif action == "StealthScan":
            self.state["open_ports"] = stealth_scan(self.target)
            if not self.state["open_ports"]:
                self.tracker.log_result("timeout")
            return 0.8 if self.state["open_ports"] else 0.0

        elif action == "BannerGrab":
            banners = banner_grabber(self.target, self.state["open_ports"])
            self.state["services"].update(fingerprint_services(banners))
            if not banners:
                self.tracker.log_result("decoy")
            return 1.0 if self.state["services"] else 0.0

        elif action == "HTTPProbe":
            return 0.7 if 80 in self.state["open_ports"] else 0.0

        elif action == "LateralProbeSSH":
            return 0.6 if simulate_lateral_movement("SSH") else 0.0

        elif action == "LateralProbeSMB":
            return 0.5 if simulate_lateral_movement("SMB") else 0.0

        elif action == "EnumerateFiles":
            return 0.4 if simulate_file_enumeration() else 0.0

        elif action == "CredTheft":
            return 0.9 if simulate_credential_theft() else 0.0

        elif action == "ExfilTestToken":
            return 0.3 if simulate_exfil() else 0.0

        return 0.0

# ------------------- Run Loop -------------------
def run_engine():
    engine = DecisionEngine(personality="silent")

    # Define layered phases for AttackGraph
    layers = [
        ["FastScan","StealthScan"],
        ["BannerGrab","HTTPProbe"],
        ["LateralProbeSSH","LateralProbeSMB","EnumerateFiles","CredTheft"],
        ["ExfilTestToken"]
    ]

    for epoch in range(3):   # run multiple episodes
        print(f"\nEpoch {epoch+1}")

        # Use AttackGraph to suggest path
        path, score = engine.graph.best_path_layered(layers, engine.state["services"])
        print("Suggested path:", path, "score:", round(score,2))

        actions_taken = []
        rewards = []

        # Execute each phase
        for phase in ["Recon","Fingerprinting","PostEx","Exfil"]:
            engine.run_phase(phase)
            if engine.state["last_action"]:
                actions_taken.append(engine.state["last_action"][0])
                rewards.append(engine.policy.Q[engine.state["last_action"][0]])

        # Log epoch results
        log_epoch(epoch+1, actions_taken, rewards, engine.state["reactions"])

        time.sleep(0.2)

    # Save learned policy and graph
    save_policy(engine.policy)
    engine.graph.save()

    print("\nLearned Q-values:")
    for a, q in sorted(engine.policy.Q.items(), key=lambda x: -x[1]):
        print(f"- {a}: {q:.2f}")

if __name__ == "__main__":
    run_engine()