import pinocchio as pin
import numpy as np

URDF = "/home/kye/mujoco_learning/basic/scenes/dual_arm/dual_panda.urdf"
full = pin.buildModelFromUrdf(URDF)

fingers = [n for n in full.names if "finger_joint" in n]
lock_ids = [full.getJointId(n) for n in fingers]
model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
data = model.createData()
assert model.nv == 14

L, R = "panda_1_hand_tcp", "panda_2_hand_tcp"
for n in (L, R):
    assert model.existFrame(n), f"no frame {n}"
fid_L, fid_R = model.getFrameId(L), model.getFrameId(R)

q = pin.randomConfiguration(model)
pin.computeJointJacobians(model, data, q)
pin.updateFramePlacements(model, data)

J_L = pin.getFrameJacobian(model, data, fid_L, pin.LOCAL_WORLD_ALIGNED)  # 6×14
J_R = pin.getFrameJacobian(model, data, fid_R, pin.LOCAL_WORLD_ALIGNED)

print(f"left jaco shape: {J_L.shape} | right jaco shape: {J_R.shape}")

# augment jacobian
J_aug = np.vstack([J_L, J_R])
print(f"augmented jaco shape: {J_aug.shape}")

# rank
U, S, Vt = np.linalg.svd(J_aug)
rank = np.sum(S > 1e-16)
print(f"rank of J aug is: {rank} | dim of nullspace is {14 - rank}")

# damped pseudoinverse
lam = 0.05
J_pinv = J_aug.T @ np.linalg.inv(J_aug @ J_aug.T + lam**2 * np.eye(12))
N = np.eye(14) - J_pinv @ J_aug

dq_null = N @ np.random.randn(14)
V_aug = J_aug @ dq_null
print(f"‖J_aug · q̇_filtered‖ = {np.linalg.norm(V_aug):.2e}")

# strict damped pseudoinverse
J_mp = np.linalg.pinv(J_aug, rcond=1e-6)
P_mp = np.eye(14) - J_mp @ J_aug
print(f"‖J_aug · P_mp‖ = {np.linalg.norm(J_aug @ P_mp):.2e}")