import streamlit as st
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
VIRTUAL_FIELD_DIR = "Virtual field"
DRONE_PARK_DIR = "Drone park"
GRID_SIZE = 8
SECTORS_TOTAL = GRID_SIZE * GRID_SIZE
COST_PER_SECTOR = 2
FIELD_FILE = "field_data.csv"
DRONE_FILE = "drone_data.csv"

# ---------------------------------------------------------------------------
# Классы (те же самые из вашего ноутбука)
# ---------------------------------------------------------------------------
class Drone:
    def __init__(self, drone_id: int, name: str, battery: float = 100.0,
                 status: str = "на базе"):
        self.id = drone_id
        self.name = name
        self.battery = battery
        self.status = status

    def save(self):
        path = os.path.join(DRONE_PARK_DIR, f"drone_{self.id}", DRONE_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "battery", "status"])
            writer.writerow([self.id, self.name, self.battery, self.status])

    @staticmethod
    def load(drone_id: int) -> "Drone":
        path = os.path.join(DRONE_PARK_DIR, f"drone_{drone_id}", DRONE_FILE)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            return Drone(int(row["id"]), row["name"],
                         float(row["battery"]), row["status"])

    def charge(self):
        self.battery = 100.0
        self.status = "на базе"
        self.save()

class Field:
    def __init__(self, field_id: int, name: str):
        self.id = field_id
        self.name = name
        self.area_ha = 0.0
        self.scanned = False
        self.grid = None

    def _dir(self):
        return os.path.join(VIRTUAL_FIELD_DIR, f"field_{self.id}")

    def scan(self):
        self.area_ha = round(random.uniform(5.0, 80.0), 2)
        temp = np.random.uniform(10.0, 35.0, (GRID_SIZE, GRID_SIZE))
        hum = np.random.uniform(20.0, 90.0, (GRID_SIZE, GRID_SIZE))
        pests = (np.random.rand(GRID_SIZE, GRID_SIZE) < 0.20).astype(int)
        self.grid = np.stack([temp, hum, pests], axis=-1)
        self.scanned = True
        self.save()

    def save(self):
        os.makedirs(self._dir(), exist_ok=True)
        path = os.path.join(self._dir(), FIELD_FILE)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["area_ha", "scanned"])
            writer.writerow([self.area_ha, int(self.scanned)])
            writer.writerow([])
            writer.writerow(["row", "col", "temperature", "humidity", "pests"])
            if self.grid is not None:
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        t, h, p = self.grid[r, c]
                        writer.writerow([r, c, f"{t:.2f}", f"{h:.2f}", int(p)])

    @staticmethod
    def load(field_id: int, name: str) -> "Field":
        field = Field(field_id, name)
        path = os.path.join(field._dir(), FIELD_FILE)
        if not os.path.exists(path):
            return field
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            meta = next(reader)
            field.area_ha = float(meta[0])
            field.scanned = bool(int(meta[1]))
            next(reader)
            next(reader)
            grid = np.zeros((GRID_SIZE, GRID_SIZE, 3))
            for row in reader:
                if len(row) < 5:
                    continue
                r, c = int(row[0]), int(row[1])
                grid[r, c] = [float(row[2]), float(row[3]), int(row[4])]
            field.grid = grid
        return field

    def apply_treatment(self, sectors_to_treat: list):
        for (r, c) in sectors_to_treat:
            self.grid[r, c, 2] = 0
            self.grid[r, c, 0] += random.uniform(-1.5, 1.5)
            self.grid[r, c, 1] += random.uniform(-5.0, 5.0)
            self.grid[r, c, 0] = np.clip(self.grid[r, c, 0], 5.0, 40.0)
            self.grid[r, c, 1] = np.clip(self.grid[r, c, 1], 10.0, 100.0)
        self.save()

    def visualize(self):
        if not self.scanned or self.grid is None:
            return None
        temp = self.grid[:, :, 0]
        hum = self.grid[:, :, 1]
        pests = self.grid[:, :, 2]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Поле: {self.name} (ID {self.id}) | Площадь: {self.area_ha} га", fontsize=14)
        im1 = axes[0].imshow(temp, cmap="YlOrRd", vmin=5, vmax=40)
        axes[0].set_title("Температура, °C")
        fig.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(hum, cmap="YlGnBu", vmin=10, vmax=100)
        axes[1].set_title("Влажность, %")
        fig.colorbar(im2, ax=axes[1])
        cmap_pests = LinearSegmentedColormap.from_list("pests", ["#e8f5e9", "#c62828"])
        im3 = axes[2].imshow(pests, cmap=cmap_pests, vmin=0, vmax=1)
        axes[2].set_title("Очаги вредителей")
        fig.colorbar(im3, ax=axes[2], ticks=[0, 1])
        for ax in axes:
            ax.set_xticks(range(GRID_SIZE))
            ax.set_yticks(range(GRID_SIZE))
        plt.tight_layout()
        return fig

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(VIRTUAL_FIELD_DIR, exist_ok=True)
    os.makedirs(DRONE_PARK_DIR, exist_ok=True)

def list_fields() -> list:
    result = []
    idx_path = os.path.join(VIRTUAL_FIELD_DIR, "index.csv")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result.append((int(row["id"]), row["name"]))
    return result

def list_drones() -> list:
    result = []
    idx_path = os.path.join(DRONE_PARK_DIR, "index.csv")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result.append((int(row["id"]), row["name"]))
    return result

def append_field_index(field_id: int, name: str):
    idx_path = os.path.join(VIRTUAL_FIELD_DIR, "index.csv")
    new_file = not os.path.exists(idx_path)
    with open(idx_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "name"])
        writer.writerow([field_id, name])

def append_drone_index(drone_id: int, name: str):
    idx_path = os.path.join(DRONE_PARK_DIR, "index.csv")
    new_file = not os.path.exists(idx_path)
    with open(idx_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "name"])
        writer.writerow([drone_id, name])

def next_id(existing: list) -> int:
    if not existing:
        return 1
    return max(x[0] for x in existing) + 1

def remove_field(field_id: int):
    """Удаляет поле и его папку, обновляет index.csv"""
    field_dir = os.path.join(VIRTUAL_FIELD_DIR, f"field_{field_id}")
    if os.path.exists(field_dir):
        import shutil
        shutil.rmtree(field_dir)
    
    # Обновляем index.csv - перезаписываем без удаленной записи
    idx_path = os.path.join(VIRTUAL_FIELD_DIR, "index.csv")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if int(row["id"]) != field_id]
        
        with open(idx_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            for row in rows:
                writer.writerow([row["id"], row["name"]])

#УДАЛЕНИЕ ПОЛЕЙ
def remove_drone(drone_id: int) -> tuple[bool, str]:
    """
    Удаляет дрона и его папку, обновляет index.csv.
    Возвращает (успех, сообщение).
    """
    drone = Drone.load(drone_id)
    if drone.status == "в работе":
        return False, "❌ Нельзя удалить дрона, который находится в работе!"
    
    drone_dir = os.path.join(DRONE_PARK_DIR, f"drone_{drone_id}")
    if os.path.exists(drone_dir):
        import shutil
        shutil.rmtree(drone_dir)
    
    # Обновляем index.csv
    idx_path = os.path.join(DRONE_PARK_DIR, "index.csv")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if int(row["id"]) != drone_id]
        
        with open(idx_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            for row in rows:
                writer.writerow([row["id"], row["name"]])
    
    return True, f"✅ Дрон '{drone.name}' удален!"

# ---------------------------------------------------------------------------
# ИНТЕРФЕЙС STREAMLIT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="🚜 Агродроны", layout="wide")
ensure_dirs()

st.title("🚜 Управление сельскохозяйственными дронами")
st.markdown("---")

# Боковое меню
st.sidebar.header("Меню")
page = st.sidebar.radio("Выберите раздел:", 
    ["📋 Главная", "🌾 Управление полями", "🚁 Парк дронов", "💧 Обработка полей"])

# ---------------------------------------------------------------------------
# ГЛАВНАЯ СТРАНИЦА
# ---------------------------------------------------------------------------
if page == "📋 Главная":
    st.header("Статус хозяйства")
    
    col1, col2, col3 = st.columns(3)
    
    fields = list_fields()
    drones = list_drones()
    
    with col1:
        st.metric("Всего полей", len(fields))
    with col2:
        st.metric("Дронов в парке", len(drones))
    with col3:
        active_drones = sum(1 for did, _ in drones if Drone.load(did).status == "в работе")
        st.metric("Дронов в работе", active_drones)
    
    st.markdown("### Быстрые действия:")
    st.info("👈 Используйте меню слева для навигации")

# ---------------------------------------------------------------------------
# УПРАВЛЕНИЕ ПОЛЯМИ
# ---------------------------------------------------------------------------
elif page == "🌾 Управление полями":
    st.header("🌾 Управление полями")
    
    tab1, tab2 = st.tabs(["➕ Добавить поле", "📊 Просмотр полей"])
    
    with tab1:
        st.subheader("Создание нового поля")
        new_field_name = st.text_input("Введите имя поля:", key="new_field")
        if st.button("Создать поле", type="primary"):
            if new_field_name.strip():
                fields = list_fields()
                fid = next_id(fields)
                os.makedirs(os.path.join(VIRTUAL_FIELD_DIR, f"field_{fid}"), exist_ok=True)
                append_field_index(fid, new_field_name)
                st.success(f"✅ Поле '{new_field_name}' (ID {fid}) создано!")
                st.rerun()
            else:
                st.error("❌ Введите имя поля!")
    with tab2:
        st.subheader("Список полей")
        fields = list_fields()
        if fields:
            for fid, name in fields:
                with st.expander(f"📍 Поле: {name} (ID: {fid})"):
                    field = Field.load(fid, name)
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Площадь:** {field.area_ha} га")
                        st.write(f"**Статус:** {'✅ Просканировано' if field.scanned else ' Не сканировалось'}")
                    
                    with col2:
                        if st.button(" Сканировать", key=f"scan_{fid}"):
                            field.scan()
                            st.success("Поле просканировано!")
                            st.rerun()
                        
                        if field.scanned:
                            if st.button("🗺️ Показать карту", key=f"map_{fid}"):
                                fig = field.visualize()
                                if fig:
                                    st.pyplot(fig)
                    
                    with col3:
                        if st.button("🗑️ Удалить", key=f"del_{fid}", type="secondary"):
                            remove_field(fid)
                            st.success(f"Поле '{name}' удалено!")
                            st.rerun()
        else:
            st.warning("Полей пока нет. Создайте первое поле!")
   

# ---------------------------------------------------------------------------
# ПАРК ДРОНОВ
# ---------------------------------------------------------------------------
elif page == "🚁 Парк дронов":
    st.header("🚁 Парк дронов")
    
    tab1, tab2 = st.tabs(["➕ Добавить дрон", "📋 Список дронов"])
    
    with tab1:
        st.subheader("Добавление нового дрона")
        new_drone_name = st.text_input("Введите имя дрона:", key="new_drone")
        if st.button("Добавить дрон", type="primary"):
            if new_drone_name.strip():
                drones = list_drones()
                did = next_id(drones)
                os.makedirs(os.path.join(DRONE_PARK_DIR, f"drone_{did}"), exist_ok=True)
                append_drone_index(did, new_drone_name)
                Drone(did, new_drone_name).save()
                st.success(f"✅ Дрон '{new_drone_name}' (ID {did}) добавлен!")
                st.rerun()
            else:
                st.error("❌ Введите имя дрона!")
    
    with tab2:
        st.subheader("Статус дронов")
        drones = list_drones()
        if drones:
            data = []
            for did, name in drones:
                drone = Drone.load(did)
                status_icon = "🔋" if drone.status == "на базе" else "✈️"
                data.append({
                    "ID": drone.id,
                    "Имя": drone.name,
                    "Заряд": f"{drone.battery:.1f}%",
                    "Статус": f"{status_icon} {drone.status}"
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.warning("Дронов пока нет. Добавьте первого дрона!")

# ---------------------------------------------------------------------------
# ОБРАБОТКА ПОЛЕЙ
# ---------------------------------------------------------------------------
elif page == "💧 Обработка полей":
    st.header("💧 Обработка полей")
    
    fields = list_fields()
    drones = list_drones()
    
    if not fields:
        st.error("❌ Нет полей. Сначала создайте поле!")
        st.stop()
    if not drones:
        st.error("❌ Нет дронов. Сначала добавьте дрона!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        field_opts = {f"{fid}: {name}": (fid, name) for fid, name in fields}
        selected_field = st.selectbox("Выберите поле:", list(field_opts.keys()))
        fid, fname = field_opts[selected_field]
        
        treatment_type = st.radio("Тип обработки:", 
            ["Полная (все секторы)", "Выборочная (только вредители)"])
    
    with col2:
        drone_opts = {f"{did}: {name}": (did, name) for did, name in drones}
        selected_drone = st.selectbox("Выберите дрона:", list(drone_opts.keys()))
        did, dname = drone_opts[selected_drone]
    
    if st.button("🚀 Запустить обработку", type="primary"):
        field = Field.load(fid, fname)
        drone = Drone.load(did)
        
        if not field.scanned:
            st.error("❌ Поле не просканировано!")
        elif drone.status == "в работе":
            st.error("❌ Дрон уже в работе!")
        else:
            pests = field.grid[:, :, 2]
            if treatment_type.startswith("Полная"):
                targets = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
            else:
                targets = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if pests[r, c] == 1]
            
            if not targets:
                st.info("ℹ️ Очагов вредителей не обнаружено. Обработка не требуется.")
            else:
                drone.status = "в работе"
                drone.save()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total = len(targets)
                processed = 0
                idx = 0
                
                while idx < total:
                    available = int(drone.battery // COST_PER_SECTOR)
                    if available == 0:
                        status_text.text("⚡ Дрон разрядился. Возврат на базу...")
                        drone.charge()
                        status_text.text("🔋 Дрон заряжен. Продолжаем...")
                        available = int(drone.battery // COST_PER_SECTOR)
                    
                    batch = min(available, total - idx)
                    for _ in range(batch):
                        drone.battery -= COST_PER_SECTOR
                        idx += 1
                        processed += 1
                    
                    drone.save()
                    progress_bar.progress(processed / total)
                    status_text.text(f"Обработано: {processed}/{total} секторов. Заряд: {drone.battery:.1f}%")
                
                field.apply_treatment(targets)
                drone.status = "на базе"
                drone.save()
                
                st.success(f"✅ Обработка поля '{fname}' завершена! Дрон '{dname}' вернулся на базу.")
                st.balloons()

# Footer
st.markdown("---")
st.caption("Система управления агродронами v1.0")
