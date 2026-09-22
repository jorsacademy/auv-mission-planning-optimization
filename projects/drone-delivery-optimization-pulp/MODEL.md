# Mathematical Model

## Sets and Indices

- `W`: set of warehouses, indexed by `w`.
- `D`: set of drones, indexed by `d`.
- `C`: set of customers, indexed by `c`, `i`, and `j`.
- `A`: set of feasible directed arcs. The model permits warehouse-to-customer, customer-to-customer, and customer-to-warehouse arcs.

## Parameters

- `q_c`: demand of customer `c`.
- `Q_d`: payload capacity of drone `d`.
- `tau_ij`: travel time from node `i` to node `j`.
- `s_c`: service time at customer `c`.
- `[e_c, l_c]`: service-time window of customer `c`.
- `T_d`: maximum route duration of drone `d`.
- `k_d`: operating cost per unit of travel time for drone `d`.
- `f_wd`: fixed deployment cost when drone `d` is deployed from warehouse `w`.
- `M`: valid big-M value for time-propagation constraints.

## Decision Variables

- `z_wd = 1` if drone `d` is deployed from warehouse `w`; 0 otherwise.
- `a_d = 1` if drone `d` is active; 0 otherwise.
- `y_dc = 1` if drone `d` serves customer `c`; 0 otherwise.
- `x_dij = 1` if drone `d` directly travels from node `i` to node `j`; 0 otherwise.
- `t_dc >= 0`: service-start time of drone `d` at customer `c`.
- `u_dc >= 0`: cumulative delivered payload after drone `d` visits customer `c`.

## Objective Function

Minimize total travel operating cost and fixed deployment cost:

$$
\min Z = \sum_{d \in D}\sum_{(i,j) \in A} k_d\tau_{ij}x_{dij}
+ \sum_{w \in W}\sum_{d \in D} f_{wd}z_{wd}
$$

## Customer Coverage

Every customer must be served exactly once:

$$
\sum_{d \in D} y_{dc} = 1 \qquad \forall c \in C
$$

## Drone Activation and Warehouse Selection

An active drone is assigned to exactly one warehouse, while an inactive drone is assigned to none:

$$
\sum_{w \in W} z_{wd} = a_d \qquad \forall d \in D
$$

## Departure from the Selected Warehouse

If drone `d` is deployed from warehouse `w`, exactly one route arc must leave that warehouse:

$$
\sum_{c \in C} x_{dwc} = z_{wd}
\qquad \forall w \in W,\ d \in D
$$

## Return to the Selected Warehouse

If drone `d` is deployed from warehouse `w`, exactly one route arc must return to that warehouse:

$$
\sum_{c \in C} x_{dcw} = z_{wd}
\qquad \forall w \in W,\ d \in D
$$

## Customer Flow Conservation

A customer assigned to a drone must have exactly one incoming route arc:

$$
\sum_{w \in W}x_{dwc}
+ \sum_{i \in C: i \neq c}x_{dic}
= y_{dc}
\qquad \forall d \in D,\ c \in C
$$

It must also have exactly one outgoing route arc:

$$
\sum_{w \in W}x_{dcw}
+ \sum_{j \in C: j \neq c}x_{dcj}
= y_{dc}
\qquad \forall d \in D,\ c \in C
$$

## Payload Capacity

Total customer demand assigned to a drone cannot exceed its payload capacity:

$$
\sum_{c \in C} q_c y_{dc} \le Q_d a_d
\qquad \forall d \in D
$$

## Route-Duration Constraint

Travel time plus customer service time cannot exceed the drone's route-duration limit:

$$
\sum_{(i,j) \in A}\tau_{ij}x_{dij}
+ \sum_{c \in C}s_cy_{dc}
\le T_da_d
\qquad \forall d \in D
$$

## Customer Time Windows

The service-start time is restricted to the customer's valid time window when that customer is assigned to the drone:

$$
e_c y_{dc} \le t_{dc} \le l_c y_{dc}
\qquad \forall d \in D,\ c \in C
$$

## Time Propagation from a Warehouse

If drone `d` flies directly from warehouse `w` to customer `c`, the service-start time at `c` must account for that flight time:

$$
t_{dc} \ge \tau_{wc} - M(1-x_{dwc})
\qquad \forall d \in D,\ w \in W,\ c \in C
$$

## Time Propagation Between Customers

If customer `j` is visited immediately after customer `i`, service at `j` cannot begin before service at `i` is completed and the drone travels to `j`:

$$
t_{dj} \ge t_{di} + s_i + \tau_{ij} - M(1-x_{dij})
$$

$$
\forall d \in D,\ i,j \in C,\ i \neq j
$$

## Return-to-Warehouse Deadline

If customer `c` is the last customer before returning to warehouse `w`, the drone must complete the route before its route-duration limit:

$$
t_{dc} + s_c + \tau_{cw}
\le T_d + M(1-x_{dcw})
$$

$$
\forall d \in D,\ c \in C,\ w \in W
$$

## Load-Based MTZ Bounds

The cumulative-load variable is active only when the customer is assigned to the drone:

$$
q_c y_{dc} \le u_{dc} \le Q_d y_{dc}
\qquad \forall d \in D,\ c \in C
$$

## Load-Based MTZ Subtour Elimination

For every used customer-to-customer arc, cumulative load must increase by the demand of the next customer:

$$
u_{dj} \ge u_{di} + q_j - M_d^{L}(1-x_{dij})
$$

$$
\forall d \in D,\ i,j \in C,\ i \neq j
$$

where `M_d^L` is a drone-specific load big-M value. In the implementation it is set to `Q_d + max_c q_c`, which safely relaxes the constraint when the arc is not selected.

Because all customer demands are strictly positive, these load-propagation constraints prevent disconnected customer-only cycles.

## Variable Domains

$$
x_{dij}, y_{dc}, z_{wd}, a_d \in \{0,1\}
$$

$$
t_{dc} \ge 0, \qquad u_{dc} \ge 0
$$

## Interpretation

The complete formulation is a multi-depot capacitated vehicle-routing problem with time windows adapted to a drone-delivery setting. Warehouse assignment, drone activation, customer assignment, route sequencing, payload feasibility, time feasibility, and subtour elimination are solved simultaneously by the MILP solver.
