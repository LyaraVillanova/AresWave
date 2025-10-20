from areswave.faultdiagram import plot_and_save_fault

output_dir = "/home/lyara/areswave/figs"

plot_and_save_fault(strike=119.91, dip=21.49, rake=0.56, depth=22.91, cost=0.0623,
                   model_name="S0167b_TAYAK", output_dir=output_dir)

plot_and_save_fault(strike=231.21, dip=76.63, rake=-5.06, depth=16.84, cost=0.0483,
                   model_name="S0167b_Model_X", output_dir=output_dir)

plot_and_save_fault(strike=185.95, dip=30.65, rake=124.27, depth=19.41, cost=0.0825,
                   model_name="S0185a_TAYAK", output_dir=output_dir)

plot_and_save_fault(strike=305.89, dip=80.44, rake=16.27, depth=30.37, cost=0.0650,
                   model_name="S0185a_Model_X", output_dir=output_dir)

plot_and_save_fault(strike=179.33, dip=61.20, rake=-68.55, depth=30.36, cost=0.0602,
                   model_name="S0234c_TAYAK", output_dir=output_dir)

plot_and_save_fault(strike=40.73, dip=52.46, rake=0.58, depth=38.46, cost=0.0546,
                   model_name="S0234c_Model_X", output_dir=output_dir)

plot_and_save_fault(strike=140.12, dip=25.95, rake=-149.64, depth=22.44, cost=0.0265,
                   model_name="S1102a_TAYAK", output_dir=output_dir)

plot_and_save_fault(strike=218.19, dip=9.03, rake=10.82, depth=32.15, cost=0.0264,
                   model_name="S1102a_Model_X", output_dir=output_dir)

plot_and_save_fault(strike=243.96, dip=27.00, rake=-138.14, depth=29.15, cost=0.1865,
                   model_name="S1153a_TAYAK", output_dir=output_dir)

plot_and_save_fault(strike=48.72, dip=57.05, rake=157.23, depth=45.66, cost=0.1894,
                   model_name="S1153a_Model_X", output_dir=output_dir)

plot_and_save_fault(strike=252.67, dip=73.83, rake=161.46, depth=52.06, cost=0.1561,
                   model_name="S1415a_TAYAK", output_dir=output_dir)

plot_and_save_fault(strike=256.43, dip=47.58, rake=72.21, depth=59.85, cost=0.1469,
                   model_name="S1415a_Model_X", output_dir=output_dir)


