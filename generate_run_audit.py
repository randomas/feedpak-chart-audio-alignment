import argparse
import audit_reports as ar
p=argparse.ArgumentParser();p.add_argument('run_root');a=p.parse_args()
out=ar.write_comparison_audit(a.run_root);print(out)
