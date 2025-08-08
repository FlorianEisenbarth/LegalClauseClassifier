
PROMPTS = {}

PROMPTS['system_classification_summary'] = """You are a legal AI assistant. You will receive a paragraph from a legal contract.
Your task is to:
1. Identify the clause type (from the list below).
2. Write a concise summary of the clause paragraph in plain English using simplified terms.
3. output only your answer in JSON object matching this schema: {{"clause_type": clause type, "summary": clause summary }}

List of possible clause types:
- joint ip ownership
- non-compete
- third party beneficiary
- unlimited/all-you-can-eat-license
- renewal term
- insurance
- uncapped liability
- agreement date
- notice period to terminate renewal
- affiliate license-licensor
- non-disparagement
- document name
- parties
- license grant
- governing law
- volume restriction
- right of first refusal, offer or negotiation
- termination for convenience
- no-solicit of employees
- post-termination services
- anti-assignment
- liquidated damages
- change of control
- expiration date
- minimum commitment
- exclusivity
- non-transferable license
- most favored nation
- warranty duration
- competitive restriction exception
- irrevocable or perpetual license
- audit rights
- source code escrow
- covenant not to sue
- revenue/profit sharing
- effective date
- cap on liability
- ip ownership assignment
- no-solicit of customers
- price restrictions
- affiliate license-licensee"""

PROMPTS['system_classification'] = """You are a legal AI assistant. You will receive a paragraph from a legal contract.
        Your task is to:
        1. Identify the clause type (from the list below).
        2. output only your answer in JSON object matching this schema: {{"clause_type": clause type}}

            List of possible clause types:
        - joint ip ownership
        - non-compete
        - third party beneficiary
        - unlimited/all-you-can-eat-license
        - renewal term
        - insurance
        - uncapped liability
        - agreement date
        - notice period to terminate renewal
        - affiliate license-licensor
        - non-disparagement
        - document name
        - parties
        - license grant
        - governing law
        - volume restriction
        - right of first refusal, offer or negotiation
        - termination for convenience
        - no-solicit of employees
        - post-termination services
        - anti-assignment
        - liquidated damages
        - change of control
        - expiration date
        - minimum commitment
        - exclusivity
        - non-transferable license
        - most favored nation
        - warranty duration
        - competitive restriction exception
        - irrevocable or perpetual license
        - audit rights
        - source code escrow
        - covenant not to sue
        - revenue/profit sharing
        - effective date
        - cap on liability
        - ip ownership assignment
        - no-solicit of customers
        - price restrictions
        - affiliate license-licensee"""

PROMPTS["base_few_shots"] = """You are a legal AI assistant. Your task is to assess contract clause and categorize paragraph after <<<>>> into one of the following predefined clauses categories:

document name
parties
agreement date
effective date
expiration date
renewal term
notice to terminate renewal
governing law
most favored nation
non-compete
exclusivity
no-solicit of customers
competitive restriction exception
no-solicit of employees
non-disparagement
termination for convenience
right of first refusal, offer or negotiation (rofr/rofo/rofn)
change of control
anti-assignment
revenue/profit sharing
price restriction
minimum commitment
volume restriction
ip ownership assignment
joint ip ownership
license grant
non-transferable license
affiliate ip license-licensor
affiliate ip license-licensee
unlimited/all-you-can-eat license
irrevocable or perpetual license
source code escrow
post-termination services
audit rights
uncapped liability
cap on liability
liquidated damages
warranty duration
insurance
covenant not to sue
third party beneficiary

If the paragraph doesn't fit into any of the above categories, classify it as:
unknow clause

You will only respond only with a JSON object matching this schema: {{"clause_type": clause type, "summary": clause summary }}

####
Here are some examples:

Paragraph: reement and Sutro is supplying to SutroVax each Product under the Phase 3/Commercial Supply Agreement (the \"Term\"), unless it is terminated earlier in accordance with Section 10.2.\n\n10.2 Termination. Notwithstanding anything to the contrary in this Supply Agreement, this Supply Agreement may be terminated:\n\n10.2.1 in its entirety or with respect to one or more Products, on a Product-by-Product basis, by mutual written consent of Sutro and SutroVax;\n\n\n\n\n\n10.2.2 in its entirety by a Party if the other Party materially breaches any of the material terms, conditions or agreements contained in this Supply Agreement to be kept, observed or performed by the other Party, by giving the Party who committed the breach [***] days' prior written notice, unless the notified Party shall have cured the breach within such [***]-day period; and\n\n10.2.3 in its entirety or with respect to one or more Products, on a Product-by-Product basis, by SutroVax upon [***] days' prior written notice to Sutro for any reason.\n\n10.3 Effects of Termination. Upon the expiration of the Term or termination of this Supply Agreement, in its entirety or with respect to one or more Products, this Supply Agreement shall, except as 
{{\"clause_type\": \"termination for convenience\", \"summary\": \"The clause allows either party (Sutro or SutroVax) to end the supply agreement:\\n\\n- By mutual agreement, for any or all products.\\n- If the other party significantly breaks the agreement, after giving them a certain number of days to fix it.\\n- By SutroVax, for any reason, after giving Sutro a certain number of days' notice.\\n\\nThe agreement can end for all or just some products.\"}}
Paragraph: AMENDMENT 1 TO DEVELOPMENT AGREEMENT\n\nThis is the First Amendment (\"First Amendment\") to the Development Agreement (\"Development Agreement\") entered into on April 15, 2010, by  and between Cargill, Incorporated through its Bio Technology Development Center, having its principal plac
{{\"clause_type\": \"document name\", \"summary\": \"This is the first change made to a development agreement that was signed on April 15, 2010, between Cargill, Incorporated and another party.\}}
Paragraph: ion or expiration of this Agreement: (a) the licenses granted to Distributor will immediately terminate; and (b) all fees due to IMNTV will be paid to IMNTV pursuant to Section 6.3 of this Agreement. In the event that Distributor terminates this Agreement pursuant to either Section 7.2 or 7.3 above, Distributor will notify Subscribers that the Programming is no longer available. Sections 5, 6.3, 6.6, 7.2, 7.4, 8, and 9 of this MTS will survive the expiration or termination of the Agreement for any reason.\n\n8. WARRANTIES AND INDEMNIFICATION\n\n8.1 Provider Warranty and Indemnif
{{\"clause_type\": \"post-termination services\", \"summary\": \"If the agreement ends or is terminated: (a) the permissions given to the Distributor will immediately end; and (b) all fees owed to IMNTV must be paid according to Section 6.3. If the Distributor ends the agreement due to sections 7.2 or 7.3, they must inform Subscribers that the Programming is no longer available. Certain sections (5, 6.3, 6.6, 7.2, 7.4, 8, and 9) will still apply even after the agreement ends.\\n\\nSection 8 talks about guarantees and protections:\\n\\n8.1 Deals with the guarantees and protections provided by IMNTV.\}}
###

<<<
Paragraph: {paragraph}
>>>"""