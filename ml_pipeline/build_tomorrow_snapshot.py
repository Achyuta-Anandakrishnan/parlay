import argparse

try:
    from ml_pipeline.upcoming_dashboard_service import build_and_save_snapshot
except ModuleNotFoundError:
    from upcoming_dashboard_service import build_and_save_snapshot


def main():
    parser = argparse.ArgumentParser(description="Build the upcoming NBA player-props UI snapshot.")
    parser.add_argument("--days", type=int, default=2, help="Number of upcoming days to include, starting tomorrow.")
    parser.add_argument(
        "--prop-type",
        choices=["points", "rebounds", "assists"],
        default="points",
        help="Which prop family snapshot to build.",
    )
    parser.add_argument(
        "--refresh-props",
        action="store_true",
        help="Attempt a fresh Odds API props pull before scoring. Falls back to cached props if unavailable.",
    )
    parser.add_argument(
        "--no-experimental",
        action="store_true",
        help="Skip the experimental OOF-minutes comparison fields.",
    )
    args = parser.parse_args()

    snapshot = build_and_save_snapshot(
        prop_type=args.prop_type,
        days=max(1, args.days),
        refresh_props=args.refresh_props,
        include_experimental=not args.no_experimental,
        start_offset=1,
    )

    summary = snapshot["summary"]
    print(f"generated_at: {snapshot['generated_at']}")
    print(f"games found: {summary['games_found']}")
    print(f"props found: {summary['props_found']}")
    print(f"matched props: {summary['matched_props']}")
    print(f"bookmakers shown: {summary['bookmakers']}")


if __name__ == "__main__":
    main()
