"""
**Submitted to ANAC 2024 SCML (OneShot track)**
*Authors*
        Name : Hajime Endo (20213)
        Nationality : Japan
        Email : endo@katfuji.lab.tuat.ac.jp
        Univ : Tokyo University of Agriculture and Technology

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 SCML.

I would like to extend my heartfelt gratitude to everyone involved in the organization of ANAC 2024,
as well as to all the participants who competed alongside me. Thank you.

"""

from __future__ import annotations
from negmas import SAOResponse, ResponseType, SAOState
from negmas.sao import SAONMI
from scml.oneshot import *
from scml.oneshot.ufun import *
import random
from typing import Any
from scml.oneshot import OneShotSyncAgent
from negmas import Contract
from itertools import chain, combinations

__all__ = ["DistRedistAgent"]


def distribute(q: int, n: int) -> list[int]:
    """Distributes n values over m bins with at
    least one item per bin assuming q > n"""
    # q < n 
    # Not enough  agents and resources for each bin.
    # The remaining agents are assigned randomly

    #q == n:
    # Exactly one agent per bin

    # q > n
    # Each bin starts with one agent and the remaining are assigned randomly

    # q is the number of agents and n is the number of groups
    from numpy.random import choice
    from collections import Counter

    if n == 0:
        return []

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    r = Counter(choice(n, q - n))
    return [r.get(_, 0) + 1 for _ in range(n)]


def powerset(iterable):
    # empty set and itself. 
    # If agents represent object, the power set of these object represent all possible outcomes of those agents
    # 1) Empty set: No agents are selected
    # 2) Single-Agent Sets: Subset where only one agent is selected
    # 3) Larger Sets: subsets where groups of agents are selected.
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class DistRedistAgent(OneShotSyncAgent):
    round: int
    random_p: float
    all_partners_info = dict()

    def __init__(self, initial_season="winter"):
        # random event effecting the agent's 
        self.random_p = 0.07
        super().__init__()
        self.current_season = initial_season

    def distribute_needs_randomly(self, needs, partner_ids) -> dict[str, int]:
        dist = dict()
        partners = len(partner_ids)
        # if I need nothing, end all negotiations
        if needs <= 0:
            dist.update(dict(zip(partner_ids, [0] * partners)))
            return dist

        # distribute my needs over my (remaining) partners.
        dist.update(dict(zip(partner_ids, distribute(int(1.35 * needs), partners))))
        return dist

    def distribute_by_info(self, need, partners):
        # Storing the needs (resources) that the agent wants to distribute
        # the agent will attempt to distribute amongst their partners.
        dist = dict()

        if len(partners) == 0:
            return dist

        if need <= 0:
        # If agents have no needs to distribute the function will distribute 0 to their partners that means no
        # distribution occurs
            dist.update(dict(zip(partners, [0] * len(partners))))
            return dist
# priority values
        partner_list = [
            [
                partner,
                (
                    (
                        self.all_partners_info[partner][1]
                        / self.all_partners_info[partner][0]
                    )
                    * (
                        self.all_partners_info[partner][0]
                        / (
                            self.all_partners_info[partner][2]
                            + self.all_partners_info[partner][0]
                        )
                    )
                )
                if (
                    self.all_partners_info[partner][0]
                    + self.all_partners_info[partner][2]
                )
                != 0
                and self.all_partners_info[partner][0] != 0
                else 0,
                0,
            ]
            for partner in partners
        ]
        priority_list = sorted(partner_list, key=lambda x: x[1], reverse=True)

        sum = 0
        for i in range(len(priority_list)):
            sum += priority_list[i][1]

        if sum == 0:
            sum = 1
        p = 0
        prob_list = []
        for i in range(len(priority_list)):
            prob_list.append((p, p + priority_list[i][1] / sum))
            p += priority_list[i][1] / sum

        for _ in range(need):
            r = random.random()
            for i in range(len(prob_list)):
                if prob_list[i][0] <= r < prob_list[i][1]:
                    priority_list[i][2] += 1
                    break

        # Force allocation of 1 or more
        for i in range(len(priority_list)):
            if priority_list[i][2] == 0:
                priority_list[i][2] = 1

        dist_list = []
        for k in partners:
            for l in priority_list:
                if k == l[0]:
                    dist_list.append(l[2])

        dist.update(dict(zip(partners, dist_list)))
        return dist

    def distribute_by_offers(self, need, partners, offers):
        # increment by one 
        dist = dict()

        if len(partners) == 0:
            return dist

        if need <= 0:
            dist.update(dict(zip(partners, [0] * len(partners))))
            return dist

        partner_list = [[partner, offers[partner][QUANTITY], 0] for partner in partners]
        # partner_list = [[partner,
        #                ((self.all_partners_info[partner][1]/self.all_partners_info[partner][0]) * (self.all_partners_info[partner][0]/(self.all_partners_info[partner][3] + self.all_partners_info[partner][0]))) if (self.all_partners_info[partner][0] + self.all_partners_info[partner][3]) != 0 and self.all_partners_info[partner][0] != 0 else 0,
        #                0] for partner in partners]
        priority_list = sorted(partner_list, key=lambda x: x[1], reverse=True)

        sum = 0
        for i in range(len(priority_list)):
            sum += priority_list[i][1]

        if sum == 0:
            sum = 1
        p = 0
        prob_list = []
        for i in range(len(priority_list)):
            prob_list.append((p, p + priority_list[i][1] / sum))
            p += priority_list[i][1] / sum

        for _ in range(need):
            r = random.random()
            for i in range(len(prob_list)):
                if prob_list[i][0] <= r < prob_list[i][1]:
                    priority_list[i][2] += 1
                    break

        # Force allocation of 1 or more
        for i in range(len(priority_list)):
            if priority_list[i][2] == 0:
                priority_list[i][2] = 1

        dist_list = []
        for k in partners:
            for l in priority_list:
                if k == l[0]:
                    dist_list.append(l[2])

            # distribute my needs over my (remaining) partners.
        dist.update(dict(zip(partners, dist_list)))
        return dist

    def first_proposals(self):
        if self.awi.is_first_level:
            partners = self.awi.my_consumers
            need = self.awi.needed_sales
        else:
            partners = self.awi.my_suppliers
            need = self.awi.needed_supplies

        s, p = self._step_and_price(best_price=True)

        if self.awi.current_step < 10 or random.random() < self.random_p:
            distribution = self.distribute_needs_randomly(need, partners)
        else:
            distribution = self.distribute_by_info(need, partners)

        d = {k: (q, s, p) if q > 0 else None for k, q in distribution.items()}

        return d

    def counter_all(self, offers, states):
        self.round += 1
        response = dict()

        # Created Seasons based upon certain factors: These can be adjusted or changed
        #Price_multipler: multiples the base price according to different to different seasons
        # Threshold : When an offer is made the threshold will be the minimum amount of resources needed
        # random_p : Randmoness of distributing resources

        # winter: P) Higher prices because of reduced supply T) able to accept lower prices P) Lower randomness
        # spring: P) lower price multiplier due to increased demand. T) higher thresholds due to abundant resources P) high randomness b/c of new surplus
        # summer P) Moderate price increase, due to travel T) Moderate Threshold
        # Autumn: P)  Stable prices T) Resources become more scarce P) Unpredictable shortages due to weather
        seasonal_factors = {
            "winter": {"price_multiplier": 1.2, "threshold": 10, "random_p": 0.3},
            "spring": {"price_multiplier": 0.9, "threshold": 15, "random_p": 0.5},
            "summer": {"price_multiplier": 1.1, "threshold": 12, "random_p": 0.4},
            "autumn": {"price_multiplier": 1.0, "threshold": 14, "random_p": 0.6},
        }
        # The current season
        season = self.current_season
        factors = seasonal_factors.get(season, {"price_multiplier": 1.0, "threshold": 15, "random_p": 0.5})

        # The threshold and the randmoness is based upon the season
        th = factors["threshold"]
        self.random_p = factors["random_p"]

        # Needs: How much is needed or required (supplies or sales)
        # issues: 
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # appling the price_multiplier to the base price based upon the season
            base_price = issues[UNIT_PRICE].rand()
            price = base_price * factors["price_multiplier"]

            # Generating all the parnter subsets for evaluation
            partners = {_ for _ in all_partners if _ in offers.keys()}
            plist = list(powerset(partners))

            offers_list = []
            for i, partner_ids in enumerate(plist):
                # Total amount of offered by these partners
                others = partners.difference(partner_ids)
                offered = sum(offers[p][QUANTITY] for p in partner_ids)

                offer_list = [offers[p] for p in partner_ids]
                # Evaluates the value of this subset of offers
                utility = self.ufun.from_offers(
                    tuple(offer_list),
                    tuple([True for _ in range(len(offer_list))]),
                    ignore_signed_contracts=False,
                )

                offers_list.append((offered, partner_ids, others, utility))

            sorted_offers_list = sorted(offers_list, key=lambda x: x[3], reverse=True)
            th_round = 16
            th_over = 2

            for i, (offered, partner_ids, others, utility) in enumerate(
                sorted_offers_list
            ):
                # if all offers are processed and needs have not been met:
                # Randmonly redistribute the needs based on the random probability
                if i == len(sorted_offers_list) - 1 and needs != 0:
                    if random.random() < self.random_p:
                        distribution = self.distribute_needs_randomly(
                            needs, partner_ids
                        )
                    else:
                        distribution = self.distribute_by_offers(
                            needs, partner_ids, offers
                        )

                    response.update(
                        {
                            k: SAOResponse(ResponseType.REJECT_OFFER, None)
                            if q == 0
                            else SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (q, self.awi.current_step, price),
                            )
                            for k, q in distribution.items()
                        }
                    )
                    break
                # only accept the offer if the offer exceeds the threshold
                if needs > offered:
                    if offered >= th:
                        others = list(others)

                        if len(others) == 0 and needs - offered >= th_over:
                            continue

                        if random.random() < self.random_p:
                            remaining_dist = self.distribute_needs_randomly(
                                needs - offered, others
                            )
                        else:
                            remaining_dist = self.distribute_by_offers(
                                needs - offered, others, offers
                            )

                        response |= {
                            k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                            for k in partner_ids
                        } | {
                            k: SAOResponse(
                                ResponseType.REJECT_OFFER,
                                (q, self.awi.current_step, price),
                            )
                            for k, q in remaining_dist.items()
                        }
                        break
                    else:
                        continue

                elif needs < offered:
                    if self.round <= th_round:
                        continue
                    else:
                        if offered - needs <= th_over:
                            others = list(others)
                            response |= {
                                k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                                for k in partner_ids
                            } | {
                                k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                                for k in others
                            }
                            break
                        else:
                            continue

                elif needs == offered:
                    others = list(others)
                    response |= {
                        k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                        for k in partner_ids
                    } | {
                        k: SAOResponse(ResponseType.END_NEGOTIATION, None)
                        for k in others
                    }
                    break


    def _step_and_price(self, best_price=False):
        """Returns current step and a random (or max) price"""
        s = self.awi.current_step
        seller = self.awi.is_first_level
        issues = (
            self.awi.current_output_issues if seller else self.awi.current_input_issues
        )
        pmin = issues[UNIT_PRICE].min_value
        pmax = issues[UNIT_PRICE].max_value
        if best_price:
            return s, pmax if seller else pmin
        return s, random.randint(pmin, pmax)

    def before_step(self):
        self.round = 0

        if self.awi.current_step == 0:
            if self.awi.is_first_level == True:
                all_partners = self.awi.my_consumers
            else:
                all_partners = self.awi.my_suppliers

            # (Number of successful contracts, Contract quantity, Number of contract rounds, Number of unsuccessful contracts)
            self.all_partners_info = {partner: [0, 0, 0] for partner in all_partners}

        return super().before_step()

    def on_negotiation_success(self, contract: Contract, mechanism: SAONMI) -> None:
        if self.awi.is_first_level == True:
            partner = contract.annotation["buyer"]
        else:
            partner = contract.annotation["seller"]
        self.all_partners_info[partner][0] += 1  # Number of successful contracts
        self.all_partners_info[partner][1] += contract.agreement[
            "quantity"
        ]  # Contract quantity

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: SAONMI,
        state: SAOState,
    ) -> None:
        if self.awi.is_first_level == True:
            partner = annotation["buyer"]
        else:
            partner = annotation["seller"]
        self.all_partners_info[partner][2] += (
            1  # Number of unsuccessful contracts(EndNegotiation)
        )


if __name__ == "__main__":
    import sys

    from helpers.runner import run

    run([DistRedistAgent], sys.argv[1] if len(sys.argv) > 1 else "oneshot")
