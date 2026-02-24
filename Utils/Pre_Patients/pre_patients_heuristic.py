import math
import copy

def compute_app_effectiveness(app_count, theta_type, params):
    if theta_type == 'lin':
        theta_base = params['theta_base']
        lin_increase = params['lin_increase']
        eff = theta_base + lin_increase * app_count
        return min(eff, 1)
    elif theta_type == 'exp':
        theta_base = params['theta_base']
        k = params['k_learn']
        eff = theta_base + (1 - theta_base) * (1 - math.exp(-k * app_count))
        return min(eff, 1)
    elif theta_type == 'sigmoid':
        theta_base = params['theta_base']
        k = params['k_learn']
        infl_point = params['infl_point']
        eff = theta_base + (1 - theta_base) / (1 + math.exp(-k * (app_count - infl_point)))
        return min(eff, 1)
    else:
        return params['const']

def pre_processing_schedule(P, P_F, T, D, Entry_p, CReq_p, theta_app_type, learning_params, TS, FS, Max_t_input, nr_c, T_mapping, verbose=True):

    if verbose:
        print("PRE-PATIENT SCHEDULING: FEASIBILITY PRE-CHECK")

    # Check if entry day demands exceed capacity
    from collections import defaultdict
    entry_demand = defaultdict(int)
    entry_patients = defaultdict(list)

    for p in P:
        entry_day = Entry_p[p]
        entry_demand[entry_day] += nr_c[p]  # Aggregate by profile count
        entry_patients[entry_day].append(p)

    feasible = True
    bottlenecks = []

    for d in sorted(entry_demand.keys()):
        demand = entry_demand[d]
        capacity = sum(Max_t_input.get((t, d), 0) for t in T)

        if demand > capacity:
            deficit = demand - capacity
            if verbose:
                print(f"❌ INFEASIBLE: Period {d} requires {demand} slots, only {capacity} available (deficit: {deficit})")
                print(f"   Affected patients: {entry_patients[d]}")
            feasible = False
            bottlenecks.append(d)
        else:
            slack = capacity - demand
            if verbose:
                status = "✅" if slack >= 2 else "⚠️ "
                print(f"{status} Period {d}: {demand} demand, {capacity} capacity (slack: {slack})")

    if not feasible:
        error_msg = f"Infeasible Periods: {bottlenecks}. "
        for d in bottlenecks:
            demand = entry_demand[d]
            capacity = sum(Max_t_input.get((t, d), 0) for t in T)
            error_msg += f"Period {d}: Needs {demand}, Has {capacity} (Deficit {demand-capacity}). "

        if verbose:
            print("\n" + "=" * 100)
            print("❌ PRE-PATIENT SCHEDULING IS STRUCTURALLY INFEASIBLE!")
            print("=" * 100)
            print(error_msg)
            print("=" * 100)
        return 'Infeasible', error_msg

    if verbose:
        print("✅ PRE-CHECK PASSED - PROCEEDING WITH SCHEDULING")
        print("=" * 100 + "\n")

    Max_t = copy.deepcopy(Max_t_input)
    x, y, LOS = {}, {}, {}
    cumEff_dict = {}
    therapist_assignments = {}
    therapist_load = {t: 0 for t in T}
    app_count = {p: 0 for p in P}
    LOS_total = 0

    # === Step 0: Reserve capacity for PF patients ===
    for pf in P_F:
        entry_day = Entry_p[pf]
        available_t = [t for t in T if Max_t[(t, entry_day)] >= nr_c[pf]]
        if available_t:
            t_res = available_t[0]
            Max_t[(t_res, entry_day)] -= nr_c[pf]
            if verbose:
                print(f"Reserved therapist {t_res} on day {entry_day} for PF patient {pf} (count: {nr_c[pf]})")
        else:
            if verbose:
                print(f"Infeasible: No therapist available for PF patient {pf} (count: {nr_c[pf]}) on entry day {entry_day}")
            return 'Infeasible'

    # === Step 1: Schedule each patient ===
    for p in P:
        if verbose:
            print(f"\nScheduling patient {p} with Req: {CReq_p[p]} (count: {nr_c[p]})")
        d_current = Entry_p[p]
        cum_effective = 0

        available_therapists = [(therapist_load[t], t) for t in T if Max_t[(t, d_current)] >= nr_c[p]]
        if available_therapists:
            _, assigned = min(available_therapists)
            Max_t[(assigned, d_current)] -= nr_c[p]
            therapist_load[assigned] += nr_c[p]
            if verbose:
                print('Counter', nr_c[p])
            x[(p, assigned, d_current)] = nr_c[p]
            cum_effective += 1
            therapist_assignments[p] = assigned
            if verbose:
                print(f"Day {d_current}: Therapy with therapist {assigned} (first session, count: {nr_c[p]}). CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: -")
        else:
            if verbose:
                print(f"No therapist available for first session of patient {p} (count: {nr_c[p]}) on day {d_current}")
            continue

        d_current += 1

        while cum_effective < CReq_p[p]:
            if d_current > max(D):
                if verbose:
                    print(f"Patient {p} could not be fully treated within planning horizon.")
                break

            t = therapist_assignments[p]
            theta_app_eff = compute_app_effectiveness(app_count[p], theta_app_type, learning_params)

            window_start = max(Entry_p[p], d_current - TS + 1)
            window_end = d_current
            full_window = (window_end - window_start + 1) >= TS

            if full_window:
                therapy_count = sum(1 for d in range(window_start, window_end + 1) if any((p, tt, d) in x for tt in T))
                if therapy_count < FS and Max_t[(t, d_current)] >= nr_c[p]:
                    x[(p, t, d_current)] = nr_c[p]
                    Max_t[(t, d_current)] -= nr_c[p]
                    therapist_load[t] += nr_c[p]
                    cum_effective += 1
                    if verbose:
                        print(f"Day {d_current}: Therapy (FS rule, count: {nr_c[p]}). CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: -")
                else:
                    y[(p, d_current)] = 1
                    app_count[p] += 1
                    cum_effective += theta_app_eff
                    if verbose:
                        print(f"Day {d_current}: App (FS satisfied). CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: {theta_app_eff:.2f}")
            else:
                if (CReq_p[p] - cum_effective) >= 1 and Max_t[(t, d_current)] >= nr_c[p]:
                    x[(p, t, d_current)] = nr_c[p]
                    Max_t[(t, d_current)] -= nr_c[p]
                    therapist_load[t] += nr_c[p]
                    cum_effective += 1
                    if verbose:
                        print(f"Day {d_current}: Therapy (accelerating discharge, count: {nr_c[p]}). CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: -")
                else:
                    y[(p, d_current)] = 1
                    app_count[p] += 1
                    cum_effective += theta_app_eff
                    if verbose:
                        print(f"Day {d_current}: App. CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: {theta_app_eff:.2f}")

            d_current += 1

        # === Ensure last treatment is therapy ===
        if cum_effective >= CReq_p[p]:
            last_days = [d for (pp, tt, d) in x if pp == p] + [d for (pp, d) in y if pp == p]
            last_day = max(last_days) if last_days else Entry_p[p]

            if not any((p, tt, last_day) in x for tt in T):
                if verbose:
                    print(f"Last session not therapy. Trying to switch or extend...")
                switched = False
                if Max_t[(t, last_day)] >= nr_c[p]:
                    x[(p, t, last_day)] = nr_c[p]
                    Max_t[(t, last_day)] -= nr_c[p]
                    therapist_load[t] += nr_c[p]
                    if (p, last_day) in y:
                        del y[(p, last_day)]
                        app_count[p] -= 1
                    cum_effective = cum_effective - theta_app_eff + 1
                    if verbose:
                        print(f"Day {last_day}: Last switched to therapy (count: {nr_c[p]}). CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: -")
                    switched = True
                else:
                    while not switched:
                        last_day += 1
                        if last_day > max(D):
                            if verbose:
                                print(f"Patient {p} could not be assigned final therapy within planning horizon.")
                            break
                        theta_app_eff = compute_app_effectiveness(app_count[p], theta_app_type, learning_params)
                        y[(p, last_day)] = 1
                        app_count[p] += 1
                        cum_effective += theta_app_eff
                        if verbose:
                            print(f"Day {last_day}: App added. CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: {theta_app_eff:.2f}")

                        if Max_t[(t, last_day)] >= nr_c[p]:
                            x[(p, t, last_day)] = nr_c[p]
                            Max_t[(t, last_day)] -= nr_c[p]
                            therapist_load[t] += nr_c[p]
                            if (p, last_day) in y:
                                del y[(p, last_day)]
                                app_count[p] -= 1
                            cum_effective = cum_effective - theta_app_eff + 1
                            if verbose:
                                print(f"Day {last_day}: Therapy assigned after app extension (count: {nr_c[p]}). CumEff: {cum_effective:.2f}, AppCount: {app_count[p]}, AppEff: -")
                            switched = True

            LOS[p] = last_day - Entry_p[p] + 1
            cumEff_dict[p] = cum_effective
            if verbose:
                print(f"Patient {p} LOS: {LOS[p]}, final CumEff: {cum_effective:.2f}")
            LOS_total += LOS[p]

    if verbose:
        print(f"\nGlobal total LOS: {LOS_total}")

    filtered_x = {(p, t, d): v for (p, t, d), v in x.items() if d > 0}

    return x, y, LOS, transform_t_values(filtered_x, T_mapping), filtered_x

def transform_t_values(original_dict, t_mapping):
    from collections import defaultdict

    transformed_dict = defaultdict(int)

    for (p, t, d), value in original_dict.items():
        new_t = t_mapping.get(t, t)
        new_key = (p, new_t, d)

        transformed_dict[new_key] += value

    return dict(transformed_dict)