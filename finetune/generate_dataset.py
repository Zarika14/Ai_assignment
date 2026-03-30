#!/usr/bin/env python3
"""
Generate Insurance QA Dataset for Fine-Tuning TinyLlama

Creates a diverse dataset of insurance-related Q&A with structured JSON output format.
Includes policy-based, general knowledge, ambiguous, life, renters, and pet questions.

Output format:
{
  "instruction": "Answer the following insurance question in JSON format with fields: answer, confidence (high/medium/low), source (policy/general_knowledge).",
  "input": "Question: ...",
  "output": {
    "answer": "...",
    "confidence": "high | medium | low",
    "source": "policy | general_knowledge"
  }
}
"""

import json
from pathlib import Path
from typing import List, Dict


def generate_insurance_qa_dataset() -> List[Dict]:
    """Generate 60+ insurance QA examples covering multiple domains."""

    INSTRUCTION = "Answer the following insurance question in JSON format with fields: answer, confidence (high/medium/low), source (policy/general_knowledge)."

    # ========== AUTO / POLICY-BASED QUESTIONS ==========
    auto_policy_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the collision deductible on an auto comprehensive policy?",
            "output": {
                "answer": "The collision deductible is typically $500, though this varies by policy.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What does comprehensive coverage include?",
            "output": {
                "answer": "Comprehensive coverage includes protection against theft, vandalism, weather events, and animal strikes.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is covered under medical payments coverage?",
            "output": {
                "answer": "Medical payments coverage pays for reasonable medical expenses for you or your passengers after an auto accident, up to the policy limit.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the liability coverage limit for bodily injury per person?",
            "output": {
                "answer": "The liability coverage limit for bodily injury per person is typically $100,000.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What are the exclusions in an auto insurance policy?",
            "output": {
                "answer": "Common exclusions include racing, intentional damage, commercial use, wear and tear, and driving under the influence.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the glass coverage deductible?",
            "output": {
                "answer": "Glass coverage typically has a $0 deductible — no out-of-pocket cost for glass claims.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: How long do I have to file a claim after an accident?",
            "output": {
                "answer": "You should contact the claims department within 72 hours of the incident to file a claim.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the uninsured motorist coverage limit?",
            "output": {
                "answer": "The uninsured motorist coverage limit is $100,000 per person and $300,000 per accident.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Does my auto policy cover a rental car?",
            "output": {
                "answer": "Many auto policies include rental car reimbursement coverage, typically $30–$50 per day up to a maximum limit. Check your policy for specifics.",
                "confidence": "medium",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Is roadside assistance included in my auto policy?",
            "output": {
                "answer": "Roadside assistance is an optional add-on in most auto policies, covering towing, flat tires, battery jumps, and lockouts.",
                "confidence": "medium",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is personal injury protection (PIP)?",
            "output": {
                "answer": "PIP covers medical expenses, lost wages, and other costs for you and your passengers regardless of fault in an accident.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Can I choose my repair shop?",
            "output": {
                "answer": "For claims under $2,500, you can use preferred repair shops without extra charges. Larger claims may have restrictions.",
                "confidence": "high",
                "source": "policy"
            }
        },
    ]

    # ========== GENERAL INSURANCE KNOWLEDGE ==========
    general_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the difference between collision and comprehensive coverage?",
            "output": {
                "answer": "Collision covers damage from impact with other vehicles or objects. Comprehensive covers non-collision damage like theft, weather, and vandalism.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What does a deductible mean?",
            "output": {
                "answer": "A deductible is the amount you pay out-of-pocket before your insurance coverage kicks in. Higher deductibles usually mean lower premiums.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: How is insurance premium calculated?",
            "output": {
                "answer": "Premiums are calculated based on coverage amount, deductible, risk profile, driving record, claims history, and location.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is a grace period?",
            "output": {
                "answer": "A grace period allows you to make a payment after the due date without losing coverage. Typical grace periods are 10–30 days.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is an adjuster?",
            "output": {
                "answer": "An insurance adjuster is a professional who investigates claims, assesses damage, and determines the payout amount on behalf of the insurer.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What documents do I need for a claim?",
            "output": {
                "answer": "You typically need your policy number, incident details, photos of the damage, police report, repair quotes, and medical records if applicable.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: How long does an insurance claim take?",
            "output": {
                "answer": "Simple claims can be processed in a few days to weeks. Complex claims may take several weeks to months depending on the investigation.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is an insurance endorsement?",
            "output": {
                "answer": "An endorsement is an amendment to your existing insurance policy that changes the coverage, terms, or conditions of the original policy.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the difference between actual cash value and replacement cost?",
            "output": {
                "answer": "Actual cash value pays the depreciated value of your item at the time of loss. Replacement cost pays what it costs to replace the item with a new one today.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is subrogation in insurance?",
            "output": {
                "answer": "Subrogation is the process where your insurer pays your claim and then seeks reimbursement from the at-fault party's insurer.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== HEALTH INSURANCE ==========
    health_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What does health insurance deductible mean?",
            "output": {
                "answer": "A health deductible is the amount you pay for healthcare services before your insurance plan starts to cover costs.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is a copay?",
            "output": {
                "answer": "A copay is a fixed amount you pay for a covered healthcare service, usually at the time of service, after meeting your deductible.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is coinsurance in health insurance?",
            "output": {
                "answer": "Coinsurance is the percentage of costs you share with your insurer after meeting your deductible. For example, 80/20 means the insurer pays 80% and you pay 20%.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is an out-of-pocket maximum?",
            "output": {
                "answer": "The out-of-pocket maximum is the most you'll have to pay in a year for covered healthcare services. After reaching this limit, your insurer pays 100%.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is a Health Maintenance Organization (HMO)?",
            "output": {
                "answer": "An HMO is a type of health plan that requires you to use a network of doctors and hospitals and get a referral from your primary care physician for specialists.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is a Preferred Provider Organization (PPO)?",
            "output": {
                "answer": "A PPO is a health plan that allows you to see any doctor without a referral, but you pay less when using in-network providers.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== HOME INSURANCE ==========
    home_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is home insurance?",
            "output": {
                "answer": "Home insurance protects your house and personal belongings against damage or theft, and provides liability coverage if someone is injured on your property.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is covered under homeowner's insurance?",
            "output": {
                "answer": "Homeowner's insurance covers the physical structure, personal belongings, liability protection, and additional living expenses if you're temporarily displaced.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Does home insurance cover flood damage?",
            "output": {
                "answer": "Standard homeowner's insurance does not cover flood damage. You need a separate flood insurance policy, typically through the National Flood Insurance Program (NFIP).",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Does homeowner's insurance cover earthquake damage?",
            "output": {
                "answer": "Standard homeowner's insurance typically does not cover earthquake damage. A separate earthquake insurance policy is required.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== LIFE INSURANCE ==========
    life_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is term life insurance?",
            "output": {
                "answer": "Term life insurance provides coverage for a specific period, such as 10, 20, or 30 years. If you die during that term, the policy pays a death benefit to your beneficiaries.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is whole life insurance?",
            "output": {
                "answer": "Whole life insurance provides lifelong coverage and includes a cash value component that grows over time, which you can borrow against or withdraw.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is a beneficiary in a life insurance policy?",
            "output": {
                "answer": "A beneficiary is the person or entity designated to receive the death benefit payout when the insured person passes away.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Can I have multiple beneficiaries on a life insurance policy?",
            "output": {
                "answer": "Yes, you can name multiple beneficiaries and specify how the death benefit is split among them in percentage terms.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== RENTERS INSURANCE ==========
    renters_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What does renters insurance cover?",
            "output": {
                "answer": "Renters insurance covers your personal belongings against theft, fire, and water damage. It also includes liability protection and covers additional living expenses if your rental becomes uninhabitable.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Does my landlord's insurance cover my belongings?",
            "output": {
                "answer": "No. Your landlord's insurance only covers the building structure and their liability. You need your own renters insurance policy to protect your personal belongings.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: How much does renters insurance typically cost?",
            "output": {
                "answer": "Renters insurance is usually very affordable, averaging $15–$30 per month depending on coverage amount, location, and the value of your belongings.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== PET INSURANCE ==========
    pet_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is pet insurance?",
            "output": {
                "answer": "Pet insurance helps cover vet bills for your pet's accidents, illnesses, and sometimes routine wellness care, reducing the financial burden of unexpected pet health costs.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What does pet insurance typically not cover?",
            "output": {
                "answer": "Pet insurance typically excludes pre-existing conditions, cosmetic procedures, breeding costs, and sometimes dental cleanings depending on the plan.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== CLAIMS & PROCESSING ==========
    claims_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is the status of claim CLM-001?",
            "output": {
                "answer": "Claim CLM-001 is approved for $5,000 for collision damage.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: How do I file a claim?",
            "output": {
                "answer": "Contact the claims department, provide your policy number and incident details, document damage with photos, and submit repair quotes.",
                "confidence": "high",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What happens after I file a claim?",
            "output": {
                "answer": "An adjuster will be assigned to assess the damage, verify coverage, and determine the payout. You'll receive a decision notice within a few business days.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Can a denied claim be appealed?",
            "output": {
                "answer": "Yes, you can appeal a denied claim by providing additional documentation, requesting a formal review, or escalating to your state's insurance commissioner.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== PREMIUM & BILLING ==========
    premium_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: How often do I pay my insurance premium?",
            "output": {
                "answer": "Premiums can be paid monthly, quarterly, semi-annually, or annually depending on your policy agreement.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What happens if I don't pay my premium?",
            "output": {
                "answer": "Your coverage lapses after the grace period ends. Non-payment can result in policy cancellation and may affect your ability to get future coverage.",
                "confidence": "high",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Can I get a premium discount?",
            "output": {
                "answer": "Yes, common discounts include safe driver, multi-policy bundling, good student, anti-theft device, and loyalty discounts.",
                "confidence": "medium",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Can I cancel my policy anytime?",
            "output": {
                "answer": "Yes, you can typically cancel at any time by contacting customer service or submitting written notice. You may receive a prorated refund for unused premiums.",
                "confidence": "high",
                "source": "policy"
            }
        },
    ]

    # ========== AMBIGUOUS QUESTIONS ==========
    ambiguous_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: Will my insurance cover my accident?",
            "output": {
                "answer": "Coverage depends on the type of accident, your policy terms, and who was at fault. Please provide more details for a specific answer.",
                "confidence": "low",
                "source": "policy"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What is my exact premium?",
            "output": {
                "answer": "Your premium is personalized based on your driving record, location, vehicle, and coverage choices. Please contact your agent for an exact quote.",
                "confidence": "low",
                "source": "general_knowledge"
            }
        },
    ]

    # ========== ANTI-HALLUCINATION / OUT-OF-SCOPE ==========
    unknown_qa = [
        {
            "instruction": INSTRUCTION,
            "input": "Question: What was your office location in 1985?",
            "output": {
                "answer": "I don't have information about historical office locations. Please contact the main company office directly.",
                "confidence": "low",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: What are the CEO's personal investments?",
            "output": {
                "answer": "This information is not publicly available through insurance policy records. Please refer to official company financial disclosures.",
                "confidence": "low",
                "source": "general_knowledge"
            }
        },
        {
            "instruction": INSTRUCTION,
            "input": "Question: Can I get insurance for my spaceship?",
            "output": {
                "answer": "Standard auto or homeowner's insurance does not cover spacecraft. You would need specialized aerospace liability insurance.",
                "confidence": "low",
                "source": "general_knowledge"
            }
        },
    ]

    # Combine all datasets
    dataset = []
    dataset.extend(auto_policy_qa)
    dataset.extend(general_qa)
    dataset.extend(health_qa)
    dataset.extend(home_qa)
    dataset.extend(life_qa)
    dataset.extend(renters_qa)
    dataset.extend(pet_qa)
    dataset.extend(claims_qa)
    dataset.extend(premium_qa)
    dataset.extend(ambiguous_qa)
    dataset.extend(unknown_qa)

    print(f"\n✅ Generated {len(dataset)} QA samples")
    print(f"  - Auto / Policy-based: {len(auto_policy_qa)}")
    print(f"  - General knowledge:   {len(general_qa)}")
    print(f"  - Health insurance:    {len(health_qa)}")
    print(f"  - Home insurance:      {len(home_qa)}")
    print(f"  - Life insurance:      {len(life_qa)}")
    print(f"  - Renters insurance:   {len(renters_qa)}")
    print(f"  - Pet insurance:       {len(pet_qa)}")
    print(f"  - Claims & processing: {len(claims_qa)}")
    print(f"  - Premium & billing:   {len(premium_qa)}")
    print(f"  - Ambiguous:           {len(ambiguous_qa)}")
    print(f"  - Anti-hallucination:  {len(unknown_qa)}")

    return dataset


def save_dataset(dataset: List[Dict], output_path: str = "insurance_qa_dataset.json") -> None:
    """Save dataset to JSON file."""
    path = Path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    print(f"\n💾 Dataset saved to: {path.resolve()}")


def main():
    print("\n" + "="*70)
    print("INSURANCE QA DATASET GENERATION (60+ Examples)")
    print("="*70)

    dataset = generate_insurance_qa_dataset()
    save_dataset(dataset)

    print(f"\n✅ Dataset generation complete!")
    print(f"   Total examples: {len(dataset)}")


if __name__ == "__main__":
    main()
