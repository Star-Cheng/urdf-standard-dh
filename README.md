# urdf-standard-dh

从一个 URDF 树中选择任意 `base_link → tip_link` 路径，解析构造运动学等价的**标准 DH 或 Craig MDH 参数**，同时导出基座、工具固定变换和关节变量映射。支持桌面界面、Python API 和 `urdf-dh` 命令行，无需 ROS。当前版本为 `0.4.0`，默认使用标准 DH；两种模式统一采用控制项目式建系。

v0.4.0 优先沿用 URDF 关节坐标方向，将首轴轴向固定距离保留在第一行参数，法兰后的固定链独立保存为 Tool。以 `tianji_arm.urdf` 为例，两种模式都是 `B=I`、`d1=0.1745 m`，与控制项目的尺寸归属一致。原始 URDF 中 `1.5708/3.1416` 的精度保留，因此精确提取值与使用 `pi/2、pi` 的名义预设仍可能有微小差别。

本工具在本机读取与计算，不上传 URDF。当前项目位于 Linux 路径 `/home/gym/code/agent/codex/project_0016_URDF标准DH提取工具/03_work/urdf-standard-dh`，是全局工作区 `/Users/gym/code/agent/codex/` 的 Linux 回退位置。

## 安装

需要 Python 3.10 或更新版本。Python 包依赖只有 NumPy；桌面界面使用 Python 的 Tkinter 系统组件，不需要额外的 GUI pip 包。

推荐在独立虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
```

本机项目工作目录已准备好 `.venv`，可直接激活后使用。当前系统缺少 `ensurepip`，该环境通过 `python3 -m venv --without-pip --system-site-packages .venv` 创建，并使用现有系统 pip 完成安装；其他机器通常使用上面的标准创建命令即可。

开发与测试：

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest
```

## 桌面界面

在本项目目录中运行：

```bash
./launch_gui.sh
```

启动器会优先使用项目现成的 `.venv/bin/python`，否则使用 `python3`；可从其他工作目录调用，路径中含空格也可使用。Windows 可双击 `start_gui.bat`，或在终端执行它。Windows 启动器优先使用项目的 `.venv\Scripts\python.exe`，否则使用 `py -3`，再回退到 `python`。

安装包安装完成后，也可以运行：

```bash
urdf-dh-gui
# 或
python3 -m urdf_dh.gui
```

首次启动显示空界面，不自动打开用户文件。操作步骤：

1. 点击选择文件按钮，打开本地 `.urdf` 文件。
2. 在下拉框选择 `base_link` 与 `tip_link`。
3. 在提取按钮上方的“提取模式”下拉框选择“标准 DH”或“MDH（Craig）”。
4. 执行提取与验证，查看对应参数表、基座/工具固定变换及 FK 误差。
5. 选择保存格式与路径，导出完整 JSON、DH 表 CSV 或中文 Markdown 报告。

切换模式会清空旧结果并要求重新提取；表头、变换公式和保存文件名会随模式更新。模式选择器固定在操作区，在默认与最小窗口尺寸下均可见。

同一机械臂更换法兰后的 TCP，不会改变本体参数表。坐标变换页同时显示实际安装入口、法兰出口、安装 Base 与法兰后的 Tool；必要的 DH 轴向对齐单独记录。

推荐保留 JSON 或 Markdown，以便同时保存关节映射、固定变换和验证结果。CSV 适合查看 DH 表；使用时仍需保留完整模型信息。

也可以在启动时指定文件和链端点，或打开内置示例：

```bash
./launch_gui.sh /你的路径/robot.urdf --base base_link --tip tool0
./launch_gui.sh --demo
./launch_gui.sh --demo --convention mdh

# 已安装包时
urdf-dh-gui robot.urdf --base base_link --tip tool0
python3 -m urdf_dh.gui --demo
```

文件路径与 `--base`、`--tip` 均可省略，在界面中选择即可。`--convention standard` 为标准 DH，`--convention modified` 或 `--convention mdh` 为 Craig MDH。`--demo` 只读取随源码或安装包提供的示例，不修改示例文件。

当前机器的 Tk 8.6 已可用。其他 Linux 机器若提示缺少 Tkinter，可安装系统对应的 Tk 组件，例如 Debian/Ubuntu 的 `python3-tk`；Windows/macOS 可使用包含 Tcl/Tk 的 Python 发行版。GUI 需要可用的桌面显示环境，服务器或无桌面环境可继续使用下方命令行。

## 命令行

```bash
urdf-dh robot.urdf \
  --base base_link \
  --tip tool0 \
  --convention standard \
  --output outputs/robot_dh.json \
  --csv outputs/robot_dh.csv \
  --markdown outputs/robot_dh.md \
  --samples 100 \
  --seed 0
```

也可以使用 `python3 -m urdf_dh`。输出目录会自动创建；输入文件与输出文件同路径、链接到同一文件，或多个输出路径互相冲突时会拒绝执行。已有输出文件会被替换，请选择专用结果路径。

省略 `--output` 时，完整 JSON 写入标准输出，状态信息和提示写入标准错误，因此可以直接重定向：

```bash
urdf-dh robot.urdf --base base_link --tip tool0 > robot_dh.json
urdf-dh robot.urdf --base base_link --tip tool0 --convention mdh > robot_mdh.json
```

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `INPUT.urdf` | 已展开的 URDF 文件 | 必填 |
| `--base` / `--tip` | 起点 / 终点 link 名称 | 必填 |
| `--convention` | `standard` 为标准 DH；`modified` / `mdh` 为 Craig MDH | `standard` |
| `--output` | 完整 JSON 文件；省略则写 stdout | 无 |
| `--csv` | DH 表与关节映射 CSV | 无 |
| `--markdown` | 含固定变换、映射与验证的中文报告 | 无 |
| `--samples` | 随机 FK 验证样本数，正整数 | `100` |
| `--seed` | 可复现的非负随机种子 | `0` |
| `--position-tol` | 位置误差容限，米 | `1e-7` |
| `--rotation-tol` | 旋转角误差容限，弧度 | `1e-7` |
| `--parallel-tol` | 平行/反平行轴判定的夹角正弦阈值，大于 0 且小于 1，无量纲 | `1e-12` |
| `--no-validate` | 跳过采样验证；结果保留未验证状态 | 默认验证 |

转换、输入或导出失败时退出码为 `2`，成功为 `0`。默认先完成核心 FK 验证，再导出结果；验证不通过的模型不会作为成功结果写出。多个输出文件逐个写入，并非跨文件事务；例如后一个目标不可写时，前面的文件可能已经保存。

## Python API

```python
import json

from urdf_dh import extract_dh

model = extract_dh(
    "robot.urdf",
    base_link="base_link",
    tip_link="tool0",
    convention="modified",  # standard / modified；mdh 是 modified 的别名
    validate=True,
    samples=100,
    seed=0,
    position_tolerance=1e-7,
    rotation_tolerance=1e-7,
    parallel_tolerance=1e-12,
)

for row in model.rows:
    print(row.joint_name, row.a, row.alpha, row.d_offset, row.theta_offset)

# 完整结果包含 rows、基座/工具变换、关节映射、提示和验证数据。
with open("robot_dh.json", "w", encoding="utf-8") as file:
    json.dump(model.to_dict(), file, ensure_ascii=False, indent=2)

# 用独立输入关节名计算完整 base → tip 变换。
q = {name: 0.0 for name in model.input_names}
T_base_tip = model.fk(q)

# 导出文件可独立重新加载，无需原始 URDF。
from urdf_dh import DHModel

with open("robot_dh.json", encoding="utf-8") as file:
    restored = DHModel.from_dict(json.load(file))
T_restored = restored.fk(q)
```

原有 `extract_standard_dh(...)` 入口保持兼容，新增 `extract_modified_dh(...)` 专门提取 Craig MDH；两者接受相同的选链和验证参数。`dh_matrix(theta, d, a, alpha)` 与 `mdh_matrix(theta, d, a, alpha)` 分别计算两种单行矩阵。调用 `row.transform(inputs, convention=model.convention)` 可按模型模式计算单行，默认仍为标准 DH；完整位姿优先使用 `model.fk(...)`。

DH / MDH 建系通常不唯一。本工具返回与输入模型等价的一组参数，不能保证复原厂家原始 DH 表。使用结果时，应同时保留 `base_transform`、`tool_transform`、关节顺序与变量映射，单独使用四列 DH 表可能得到错误的末端位姿。

## DH 约定与变量映射

两种模式的单行变换分别为：

```text
标准 DH：A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
Craig MDH：A_i = Rx(alpha_(i-1)) Tx(a_(i-1)) Rz(theta_i) Tz(d_i)
T_base_tip(q) = base_transform × A_1(q) × … × A_n(q) × tool_transform
```

Craig MDH 使用前一段的 `a_(i-1)`、`alpha_(i-1)`。本工具选择第一行 `a=alpha=0`，起始原点由安装入口原点在首轴上的投影确定，首轴轴向尺寸保留在第一行 `d`。中间公法线尽量朝向对应 URDF 关节坐标的 X 轴，末帧原点锚定在末关节原点，X 轴优先沿用该关节坐标。法兰后的固定几何保留在 C，参数表不随纯固定 TCP 改变。约定参考 [Robotics Toolbox Link 文档](https://www.petercorke.com/RTB/r9/html/Link.html)，具体矩阵顺序以上式为准。

标准 DH 和 MDH 使用同一套边界：先构造控制项目式 MDH `X_(i-1) Z_i(q)`，其中 `X_i=Tx(a_i) Rx(alpha_i)`；取 `X_0=X_n=I`，重组为标准行 `Z_i(q) X_i`。因此两种模式的 B/C 相同，theta/d 保持原定义，a/alpha 按各自索引重组。每个活动关节对应一行，关节方向和 mimic 映射不变，分别对照原始 URDF 验证。接近 -pi 的 theta_offset 可加整圈 2*pi 表达成接近 +180°，只改变周期代表值，不舍入几何。

JSON 使用 `convention="standard"` 或 `"modified"`，并保存 `transform_order`、`parameter_indexing`、`frame_policy="controller"`。`boundary_frames` 将两端进一步拆为：

```text
B = mount_transform × base_alignment
C = tool_alignment × flange_to_tip
```

`mount_transform` 是请求起点到首活动关节进入 link 的安装变换；`flange_to_tip` 是末活动关节离开 link 到请求终点的固定链。任意 URDF 的边界轴未必符合 DH 约束，必要的固定对齐由 `base_alignment/tool_alignment` 保留，并给出提示。只有无需对齐时，B/C 才直接等于实际安装 Base/法兰后的 Tool，不能对任意模型强制清零。

旧 JSON 保留原参数和 FK，缺少新字段时记为 `frame_policy="legacy"`，不会自动重建旧表。旧 MDH 的 `Tx(a) Rx(alpha)` 顺序仍可加载，两者因同轴而可交换。新文件重载会检查约定、公式及边界分解的一致性。CSV 保留原参数列名，并记录约定、矩阵顺序、索引和建系策略；CSV 不包含 B/C，完整复算应使用 JSON 或 Markdown。

全部长度使用米，角度使用弧度。每行保存 `a`、`alpha`、`d_offset`、`theta_offset`，以及关节变量的映射：

```text
q_DH = multiplier × q[source_name] + offset

revolute / continuous:
    theta = theta_offset + q_DH
    d = d_offset

prismatic:
    theta = theta_offset
    d = d_offset + q_DH
```

`input_names` 给出独立输入关节的顺序。`source_name` 指向独立输入；`multiplier` 与 `offset` 表达 URDF mimic 关系，包括递归组合后的映射。映射偏置 `offset` 的单位与该 DH 行的关节变量相同：旋转为弧度，移动为米。`d_offset`、`theta_offset` 是 DH 几何零位，和 mimic 的 `offset` 含义不同。

`direction` 仅记录路径沿 URDF 树边正向（`+1`）或反向（`-1`）遍历。反向运动已经通过轴方向表达，计算 `q_DH` 时**不要再次乘以 `direction`**。CSV 将该字段命名为 `traversal_direction_metadata`，以提示这一点。

`geometry_case` 记录建系时的轴线几何情况；平行、重合或相交轴也可以构造 DH，但坐标系存在额外选择自由。

## 支持范围

- 支持 `revolute`、`continuous`、`prismatic`，以及路径上的 `fixed` 关节。
- 支持沿父子关系正向、反向或跨共同祖先的树路径；按 `base → tip` 顺序输出活动关节。
- 固定关节的变换会累计保留；只有固定关节的路径也可表达。
- 支持 mimic 变量映射，独立源关节可以在选定路径之外。
- `floating` 与 `planar` 关节不属于单自由度 DH 行，选定路径包含它们时会明确拒绝。
- 要求 URDF 是有效的树；闭链约束、柔性变形与机械传动误差不由 DH 转换处理。

输入需为标准 URDF XML。Xacro 必须先展开，例如在已配置 ROS/Xacro 的环境中执行：

```bash
xacro robot.urdf.xacro -o robot.urdf
```

本工具不依赖或调用 Xacro，也不读取 mesh 文件、纹理、惯量或碰撞几何来推断关节轴；关节的 `origin`、`axis`、类型和拓扑已包含所需运动学信息。

## 验证与数值边界

默认将 DH 正运动学与直接按 URDF 路径计算的正运动学进行比较，分别检查末端位置误差和旋转角误差。JSON 中的 `validation` 保留验证数据；Markdown 同时列出 DH 表、基座/工具变换、完整关节映射和验证结果。

验证包含零输入配置与 `--samples` 个随机配置，默认共 101 组。随机范围优先采用独立输入关节的限位；缺少限位时采用工具记录的有限范围。mimic 从动关节的限位不参与约束采样，因此验证范围不是机器人可执行空间的声明。可在 `validation.input_sampling_ranges` 中查看实际输入范围。

验证同时记录 `position_resolution_estimate`：机器 epsilon 乘以 B/C 平移及各行固定几何长度总和。如果该分辨率估计已超过请求的位置容差，即使有限采样碰巧通过，也会拒绝输出成功结果。这是保守的数值适用性检查，不是严格误差上界。它用于识别近似平行轴产生的巨大 DH 偏置及相消；可依据实际精度需求选择容差，默认不会自动放宽。

验证只证明在所检验配置下，这两种表示与输入 URDF 一致。它不是实机标定，不估计装配误差，也不能替代机器人实测。

相邻轴方向夹角的正弦小于或等于 `parallel_tolerance`（默认 `1e-12`）时，按平行或反平行轴建系，以吸收浮点重建引入的微小方向误差。该近似仍须通过完整 FK 的位置与姿态容差验证。阈值保存在 `geometry_tolerances.parallel_axis_sine`；极高精度需求可调小 `--parallel-tol`，同时检查验证误差和数值条件。

近乎平行的轴，其公法线落点可能远离机构，造成很大的轴向偏置 `d`，数值条件较差；公法线的最短距离本身不一定很长。不要通过任意增大容限来掩盖失败；先检查输入精度、轴线夹角与工具提示，必要时保留 URDF 或使用 POE 等其他表示。标准 DH 的数值表现受建系选择影响，不应仅凭参数大小判断机器人本体是否异常。

## 许可

本项目以 MIT 许可发布，见 [LICENSE](LICENSE)。
