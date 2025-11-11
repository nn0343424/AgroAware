// client/src/lib/season.js
// Convert month number to Indian agricultural season

export function monthToSeason(month, state = "Karnataka") {
  // month: 0=Jan, 1=Feb, ... 11=Dec
  
  // Kharif: June–October (Southwest monsoon)
  if ([5, 6, 7, 8, 9].includes(month)) {
    return "Kharif";
  }
  
  // Rabi: November–February (Post-monsoon/Winter)
  if ([10, 11, 0, 1].includes(month)) {
    return "Rabi";
  }
  
  // Summer: March–May (Pre-monsoon)
  return "Summer";
}
